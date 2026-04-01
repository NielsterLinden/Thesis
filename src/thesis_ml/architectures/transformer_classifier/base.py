from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from thesis_ml.architectures.transformer_classifier.modules.biases import (
    MIAEncoder,
    NodewiseMassBias,
)
from thesis_ml.architectures.transformer_classifier.modules.biases.backbone import (
    PairwiseFeatureBackbone,
)
from thesis_ml.architectures.transformer_classifier.modules.biases.bias_composer import (
    BiasComposer,
    _pad_bias_for_special_tokens,
    build_bias_composer,
)
from thesis_ml.architectures.transformer_classifier.modules.embedding import build_input_embedding
from thesis_ml.architectures.transformer_classifier.modules.encoder_block import build_transformer_encoder
from thesis_ml.architectures.transformer_classifier.modules.head import build_classifier_head
from thesis_ml.architectures.transformer_classifier.modules.positional import (
    RotaryEmbedding,
    get_positional_encoding,
    parse_dim_mask,
)
from thesis_ml.architectures.transformer_classifier.modules.tokenizers.tokenizers import get_feature_map


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for event-level sequence classification.

    Supports multiple physics-informed attention bias extensions via
    ``BiasComposer``, optional nodewise-mass embedding patches via
    ``NodewiseMassBias``, and the full MIParT pipeline via ``MIAEncoder``.
    """

    def __init__(
        self,
        embedding: nn.Module,
        pos_enc: nn.Module | None,
        encoder: nn.Module,
        head: nn.Module,
        positional_space: str = "model",
        # Legacy (kept for backward compat; ignored when bias_composer is set)
        pairwise_bias_net: nn.Module | None = None,
        # New physics-informed modules
        bias_composer: BiasComposer | None = None,
        nodewise_mass: NodewiseMassBias | None = None,
        mia_encoder: MIAEncoder | None = None,
        pairwise_backbone: PairwiseFeatureBackbone | None = None,
        mia_placement: str = "prepend",
        # Sequence layout constants (needed to correctly slice physical tokens)
        use_cls_token: bool = True,
        num_met_tokens: int = 0,
    ):
        """Initialize transformer classifier.

        Parameters
        ----------
        embedding : nn.Module
            Input embedding module.
        pos_enc : nn.Module | None
            Additive positional encoding (None for rotary or no-PE configs).
        encoder : nn.Module
            Transformer encoder stack.
        head : nn.Module
            Classifier head (pooling + linear).
        positional_space : str
            ``"model"`` or ``"token"``.
        pairwise_bias_net : nn.Module | None
            Legacy pairwise bias.  Ignored when bias_composer is provided.
        bias_composer : BiasComposer | None
            Physics-informed attention bias composer (new API).
        nodewise_mass : NodewiseMassBias | None
            Nodewise invariant-mass embedding patch applied before the encoder.
        mia_encoder : MIAEncoder | None
            MIParT-style pre-encoder (computes U1 → K MIA blocks → U2).
        use_cls_token : bool
            Whether the embedding prepends a CLS token (affects slice offsets).
        num_met_tokens : int
            Number of MET/METphi tokens appended by the embedding (0 or 2).
        """
        super().__init__()
        self.embedding = embedding
        self.pos_enc = pos_enc
        self.encoder = encoder
        self.head = head
        self.positional_space = positional_space
        self.use_cls_token = use_cls_token
        self.num_met_tokens = num_met_tokens

        # Legacy path
        self.pairwise_bias_net = pairwise_bias_net

        # New physics modules
        self.bias_composer = bias_composer
        self.nodewise_mass = nodewise_mass
        self.mia_encoder = mia_encoder
        self.pairwise_backbone = pairwise_backbone
        self.mia_placement = mia_placement

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def prepare_encoder_inputs(
        self,
        *args: Any,
        mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        bool,
        torch.Tensor | None,
        torch.Tensor | None,
        Any,
    ]:
        """Run embedding through bias/MIA prep; returns tensors fed to ``self.encoder``.

        Returns
        -------
        x, mask, attention_bias, is_raw, tokens_cont, tokens_id, globals_
            ``tokens_id`` and ``globals_`` are only meaningful when ``is_raw``.
        """
        # -----------------------------------------------------------------
        # 1. Parse inputs and run embedding
        # -----------------------------------------------------------------
        if len(args) == 3:
            tokens_cont, tokens_id, globals_ = args
            x, mask = self.embedding(tokens_cont, tokens_id, globals_, mask=mask)
            is_raw = True
        elif len(args) == 2:
            tokens_cont, tokens_id = args
            globals_ = None
            x, mask = self.embedding(tokens_cont, tokens_id, mask=mask)
            is_raw = True
        elif len(args) == 1:
            integer_tokens = args[0]
            tokens_cont = None
            tokens_id = None
            globals_ = None
            x, mask = self.embedding(integer_tokens, mask=mask)
            is_raw = False
        else:
            raise ValueError(f"Expected 1, 2, or 3 arguments, got {len(args)}")

        # -----------------------------------------------------------------
        # 2. Positional encoding (model-space only; token-space is in embedding)
        # -----------------------------------------------------------------
        if self.positional_space == "model" and self.pos_enc is not None:
            x = self.pos_enc(x, mask=mask)

        # -----------------------------------------------------------------
        # 3. Compute physical-token mask slice
        #    mask after embedding: [B, T_full] where T_full = T + cls + n_met
        #    Layout: [CLS | p1...pT | MET | METphi]
        # -----------------------------------------------------------------
        attention_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None

        if is_raw and tokens_cont is not None:
            T_phys = tokens_cont.size(1)
            cls_offset = 1 if self.use_cls_token else 0

            # Correct physical-token mask (fixes the bug where mask[:, 1:] was
            # used even when include_met extended mask to [B, T+2+1])
            mask_phys = mask[:, cls_offset : cls_offset + T_phys] if mask is not None else None

            # We may need to rebuild x from parts to avoid in-place ops that
            # break autograd.  Gather the unchanged prefix and suffix first.
            x_pre = x[:, :cls_offset] if cls_offset > 0 else None
            x_suf = x[:, cls_offset + T_phys :] if self.num_met_tokens > 0 else None

            # Current physical slice (non-in-place modifications below)
            x_phys = x[:, cls_offset : cls_offset + T_phys]

            # -------------------------------------------------------------
            # 4. NodewiseMassBias: add local invariant-mass residual.
            # -------------------------------------------------------------
            if self.nodewise_mass is not None:
                node_residual = self.nodewise_mass(tokens_cont, mask=mask_phys)
                if node_residual is not None:
                    x_phys = x_phys + node_residual  # new tensor, not in-place

            # Shared pairwise backbone: compute F_ij once for MIA and BiasComposer
            F_ij, feature_to_idx = None, None
            if self.pairwise_backbone is not None:
                F_ij, _, feature_to_idx = self.pairwise_backbone(tokens_cont, mask_phys)

            # -------------------------------------------------------------
            # 5. MIAEncoder: K MIA blocks update x_phys; reduce U1 → U2.
            #    placement=prepend: run before encoder, add U2 to attention_bias.
            #    placement=append: run after encoder (see step 8).
            # -------------------------------------------------------------
            if self.mia_encoder is not None and self.mia_placement == "prepend":
                x_phys, U2 = self.mia_encoder(x_phys, tokens_cont, F_ij=F_ij, feature_to_idx=feature_to_idx, mask=mask_phys)
                if U2 is not None:
                    U2_padded = _pad_bias_for_special_tokens(U2, self.use_cls_token, self.num_met_tokens)
                    attention_bias = U2_padded

            # Reconstruct x only if x_phys was updated (nodewise or MIA prepend)
            if self.nodewise_mass is not None or (self.mia_encoder is not None and self.mia_placement == "prepend"):
                parts = [p for p in [x_pre, x_phys, x_suf] if p is not None]
                x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

            # -------------------------------------------------------------
            # 6. BiasComposer: sum all enabled pairwise bias modules.
            # -------------------------------------------------------------
            if self.bias_composer is not None:
                composer_bias = self.bias_composer(
                    tokens_cont=tokens_cont,
                    tokens_id=tokens_id,
                    mask=mask_phys if mask_phys is not None else mask,
                    globals_=globals_,
                    F_ij=F_ij,
                    feature_to_idx=feature_to_idx,
                )
                if composer_bias is not None:
                    attention_bias = composer_bias if attention_bias is None else attention_bias + composer_bias

            # -------------------------------------------------------------
            # 7. Legacy pairwise_bias_net fallback (deprecated)
            # -------------------------------------------------------------
            elif self.pairwise_bias_net is not None and self.bias_composer is None:
                from thesis_ml.architectures.transformer_classifier.modules.pairwise_bias import (
                    compute_pairwise_kinematics,
                )

                pairwise_features = compute_pairwise_kinematics(tokens_cont, mask=mask_phys)
                if pairwise_features is not None:
                    legacy_bias = self.pairwise_bias_net(pairwise_features)
                    # Pad legacy bias for CLS token (no MET support in legacy path)
                    if self.use_cls_token:
                        if legacy_bias.dim() == 4:
                            B, H, T, _ = legacy_bias.shape
                            padded = legacy_bias.new_zeros(B, H, T + 1, T + 1)
                            padded[:, :, 1:, 1:] = legacy_bias
                        else:
                            B, T, _ = legacy_bias.shape
                            padded = legacy_bias.new_zeros(B, T + 1, T + 1)
                            padded[:, 1:, 1:] = legacy_bias
                        legacy_bias = padded
                    attention_bias = legacy_bias

        return x, mask, attention_bias, is_raw, tokens_cont, tokens_id, globals_

    def forward(self, *args, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        *args
            Raw format: ``(tokens_cont [B, T, C], tokens_id [B, T])``
            Raw with globals: ``(tokens_cont, tokens_id, globals_ [B, 2])``
            Binned format: ``(integer_tokens [B, T],)``
        mask : torch.Tensor, optional
            [B, T] True=valid, False=padding.  Extended internally after
            embedding to cover CLS/MET tokens.

        Returns
        -------
        torch.Tensor
            Logits [B, n_classes].
        """
        x, mask, attention_bias, is_raw, tokens_cont, tokens_id, globals_ = self.prepare_encoder_inputs(*args, mask=mask)

        # -----------------------------------------------------------------
        # Encode and classify
        # -----------------------------------------------------------------
        x = self.encoder(x, mask=mask, attention_bias=attention_bias)

        # MIA placement=append: run MIA on encoder output's physical slice
        if is_raw and self.mia_encoder is not None and self.mia_placement == "append":
            T_phys = tokens_cont.size(1)
            cls_offset = 1 if self.use_cls_token else 0
            x_pre = x[:, :cls_offset] if cls_offset > 0 else None
            x_suf = x[:, cls_offset + T_phys :] if self.num_met_tokens > 0 else None
            x_phys = x[:, cls_offset : cls_offset + T_phys]
            mask_phys = mask[:, cls_offset : cls_offset + T_phys] if mask is not None else None
            F_ij, feature_to_idx = None, None
            if self.pairwise_backbone is not None:
                F_ij, _, feature_to_idx = self.pairwise_backbone(tokens_cont, mask_phys)
            x_phys, _ = self.mia_encoder(x_phys, tokens_cont, F_ij=F_ij, feature_to_idx=feature_to_idx, mask=mask_phys)
            parts = [p for p in [x_pre, x_phys, x_suf] if p is not None]
            x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        return self.head(x, mask=mask)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_from_config(cfg: DictConfig, meta: Mapping[str, Any]) -> nn.Module:
    """Build transformer classifier from Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with ``classifier.model.*`` keys.
    meta : Mapping[str, Any]
        Data metadata: n_tokens, token_feat_dim, has_globals, n_classes,
        vocab_size, num_types.

    Returns
    -------
    nn.Module
        Assembled TransformerClassifier.
    """
    model_cfg = cfg.classifier.model
    dim = model_cfg.dim
    num_heads = model_cfg.heads
    n_classes = meta["n_classes"]

    seq_len_tokens = meta["n_tokens"]

    include_met = bool(cfg.classifier.get("globals", {}).get("include_met", False))

    positional_space = model_cfg.get("positional_space", "model")
    if positional_space not in ("model", "token"):
        raise ValueError(f"positional_space must be 'model' or 'token', got '{positional_space}'")

    dim_mask_config = model_cfg.get("positional_dim_mask", None)

    is_binned = meta.get("vocab_size") is not None
    if is_binned:
        tokenizer_name = "binned"
        tokenizer_output_dim = dim
        id_embed_dim = 8
        cont_dim = 0
    else:
        tokenizer_name = model_cfg.tokenizer.name
        cont_dim = meta.get("token_feat_dim", 4)
        id_embed_dim = model_cfg.tokenizer.get("id_embed_dim", 8)
        if tokenizer_name == "identity":
            tokenizer_output_dim = cont_dim + id_embed_dim
        elif tokenizer_name == "raw":
            tokenizer_output_dim = cont_dim
        else:
            tokenizer_output_dim = cont_dim + id_embed_dim

    feature_map = get_feature_map(tokenizer_name, tokenizer_output_dim, id_embed_dim)
    dim_mask = None
    if dim_mask_config is not None:
        if positional_space != "token":
            raise ValueError("positional_dim_mask is only supported when positional_space='token'. " f"Got positional_space='{positional_space}'")
        dim_mask = parse_dim_mask(dim_mask_config, tokenizer_output_dim, feature_map)

    pos_enc_name = model_cfg.get("positional", "sinusoidal")
    pooling = model_cfg.get("pooling", "cls")
    use_cls_token = pooling == "cls"

    num_met_tokens = 2 if include_met and not is_binned else 0

    max_seq_length_tokens = seq_len_tokens
    extra_tokens = int(use_cls_token) + num_met_tokens
    max_seq_length_model = seq_len_tokens + extra_tokens if positional_space == "model" else seq_len_tokens

    # Attention config
    attn_cfg = model_cfg.get("attention", {})
    attention_type = str(attn_cfg.get("type", "standard"))

    rotary_emb = None
    pos_enc = None
    pos_enc_for_embedding = None

    if pos_enc_name == "rotary":
        head_dim = dim // num_heads
        # Differential attention applies RoPE to sub-heads (head_dim // 2)
        rotary_head_dim = head_dim // 2 if attention_type == "differential" else head_dim
        rotary_base = model_cfg.get("rotary", {}).get("base", 10000.0)
        rotary_emb = RotaryEmbedding(head_dim=rotary_head_dim, base=rotary_base)
    elif pos_enc_name != "none":
        if positional_space == "token":
            pos_enc_for_embedding = get_positional_encoding(
                pos_enc_name,
                tokenizer_output_dim,
                max_seq_length_tokens,
                dim_mask=dim_mask,
            )
            pos_enc = None
        else:
            pos_enc = get_positional_encoding(pos_enc_name, dim, max_seq_length_model)
            pos_enc_for_embedding = None

    # MoE config (resolve to plain dict for downstream modules)
    moe_raw = model_cfg.get("moe", {})
    if moe_raw and moe_raw.get("enabled", False):
        from omegaconf import OmegaConf

        moe_cfg: dict | None = OmegaConf.to_container(moe_raw, resolve=True)
    else:
        moe_cfg = None

    # KAN config (resolve to plain dict for downstream modules)
    kan_raw = model_cfg.get("kan", {})
    if kan_raw:
        from omegaconf import OmegaConf

        kan_cfg: dict | None = OmegaConf.to_container(kan_raw, resolve=True)
    else:
        kan_cfg = None

    # FFN type
    ffn_cfg = model_cfg.get("ffn", {})
    ffn_type = str(ffn_cfg.get("type", "standard"))

    embedding = build_input_embedding(cfg, meta, pos_enc=pos_enc_for_embedding)
    encoder = build_transformer_encoder(
        cfg,
        dim,
        rotary_emb=rotary_emb,
        moe_cfg=moe_cfg,
        use_cls_token=use_cls_token,
        ffn_type=ffn_type,
        kan_cfg=kan_cfg,
    )
    head = build_classifier_head(cfg, dim, n_classes, moe_cfg=moe_cfg, kan_cfg=kan_cfg)

    # -----------------------------------------------------------------
    # Physics-informed modules
    # -----------------------------------------------------------------

    # BiasComposer (new API; covers old attn_pairwise via backward-compat logic)
    bias_composer: BiasComposer | None = None
    if not is_binned:
        bias_composer = build_bias_composer(
            cfg=model_cfg,
            num_heads=num_heads,
            model_dim=dim,
            cont_dim=cont_dim,
            use_cls=use_cls_token,
            num_met_tokens=num_met_tokens,
            kan_cfg=kan_cfg,
        )

    # Shared pairwise feature backbone (compute F_ij once for MIA + bias modules)
    pairwise_backbone: PairwiseFeatureBackbone | None = None
    if not is_binned:
        sb_cfg = model_cfg.get("shared_backbone", {})
        needs_backbone = bias_composer is not None or model_cfg.get("mia_blocks", {}).get("enabled", False)
        if sb_cfg.get("enabled", False) and needs_backbone:
            feats = sb_cfg.get("features", "all")
            if isinstance(feats, list | tuple):
                feats = list(feats)
            pairwise_backbone = PairwiseFeatureBackbone(cont_dim=cont_dim, features=feats)

    # Legacy pairwise_bias_net (only used when bias_composer is None and old
    # attn_pairwise config is present — kept so existing checkpoints continue
    # to load).
    from thesis_ml.architectures.transformer_classifier.modules.pairwise_bias import (  # noqa: PLC0415
        PairwiseBiasNet,
    )

    pairwise_bias_net = None
    if bias_composer is None and not is_binned:
        attn_pairwise = model_cfg.get("attn_pairwise", {})
        if attn_pairwise.get("enabled", False):
            hidden_dim = attn_pairwise.get("hidden_dim", 8)
            per_head = attn_pairwise.get("per_head", False)
            pairwise_bias_net = PairwiseBiasNet(
                num_features=2,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                per_head=per_head,
            )

    # NodewiseMassBias
    nodewise_mass: NodewiseMassBias | None = None
    nm_cfg = model_cfg.get("nodewise_mass", {})
    if nm_cfg.get("enabled", False) and not is_binned:
        nodewise_mass = NodewiseMassBias(
            model_dim=dim,
            cont_dim=cont_dim,
            k_values=list(nm_cfg.get("k_values", [2, 4, 8])),
            hidden_dim=int(nm_cfg.get("hidden_dim", 64)),
        )

    # MIAEncoder
    mia_encoder: MIAEncoder | None = None
    mia_cfg = model_cfg.get("mia_blocks", {})
    if mia_cfg.get("enabled", False) and not is_binned:
        reduction_dim = int(mia_cfg.get("reduction_dim", num_heads))
        if reduction_dim != num_heads:
            import warnings

            warnings.warn(
                f"mia_blocks.reduction_dim ({reduction_dim}) != num_heads ({num_heads}). " "U2 will be broadcast across heads during bias summation.",
                stacklevel=2,
            )
        mia_encoder = MIAEncoder(
            model_dim=dim,
            cont_dim=cont_dim,
            num_blocks=int(mia_cfg.get("num_blocks", 5)),
            interaction_dim=int(mia_cfg.get("interaction_dim", 64)),
            reduction_dim=reduction_dim,
            dropout=float(mia_cfg.get("dropout", 0.0)),
        )

    mia_cfg = model_cfg.get("mia_blocks", {})
    mia_placement = str(mia_cfg.get("placement", "prepend"))
    if mia_placement not in ("prepend", "append", "interleave"):
        mia_placement = "prepend"
    if mia_placement == "interleave":
        import warnings

        warnings.warn("mia_blocks.placement=interleave not implemented; using prepend", stacklevel=2)
        mia_placement = "prepend"

    return TransformerClassifier(
        embedding=embedding,
        pos_enc=pos_enc,
        encoder=encoder,
        head=head,
        positional_space=positional_space,
        pairwise_bias_net=pairwise_bias_net,
        bias_composer=bias_composer,
        nodewise_mass=nodewise_mass,
        mia_encoder=mia_encoder,
        pairwise_backbone=pairwise_backbone,
        mia_placement=mia_placement,
        use_cls_token=use_cls_token,
        num_met_tokens=num_met_tokens,
    )
