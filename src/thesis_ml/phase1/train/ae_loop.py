from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from thesis_ml.data.h5_loader import make_dataloaders
from thesis_ml.phase1.autoenc.base import build_from_config
from thesis_ml.phase1.autoenc.losses.recon import reconstruction_loss
from thesis_ml.plots.io_utils import append_jsonl_event, append_scalars_csv
from thesis_ml.plots.orchestrator import handle_event
from thesis_ml.utils import TrainingProgressShower, build_event_payload
from thesis_ml.utils.seed import set_all_seeds

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics", "recon", "codebook", "latency"}


def _gather_meta(cfg: DictConfig, ds_meta: Mapping[str, Any]) -> None:
    # attach data-derived meta to cfg for module constructors (temporarily disable struct)
    prev_struct = OmegaConf.is_struct(cfg)
    try:
        OmegaConf.set_struct(cfg, False)
        cfg.meta = OmegaConf.create(
            {
                "n_tokens": int(ds_meta["n_tokens"]),
                "cont_dim": int(ds_meta["cont_dim"]),
                "globals": int(ds_meta["globals"]),
                "num_types": int(ds_meta["num_types"]),
            }
        )
    finally:
        OmegaConf.set_struct(cfg, prev_struct)


def train(cfg: DictConfig):
    set_all_seeds(cfg.phase1.trainer.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_dl, val_dl, test_dl, meta = make_dataloaders(cfg)
    _gather_meta(cfg, meta)

    # model assembly
    model = build_from_config(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.phase1.trainer.lr, weight_decay=cfg.phase1.trainer.weight_decay)

    # logging dir from Hydra (assumes job.chdir=true or explicit run.dir)
    outdir = None
    if cfg.logging.save_artifacts:
        outdir = os.getcwd()  # Hydra sets this when chdir=true
        os.makedirs(outdir, exist_ok=True)
        # Persist composed config beside artifacts for reproducibility
        OmegaConf.save(config=cfg, f=os.path.join(outdir, "cfg.yaml"))

    # on_start
    if outdir and cfg.logging.save_artifacts:
        start_payload = build_event_payload(
            moment="on_start",
            run_dir=outdir,
            cfg=cfg,
        )
        append_jsonl_event(str(outdir), start_payload)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_start", start_payload)

    def run_epoch(loader, do_train: bool):
        model.train(mode=do_train)
        tot = {"loss": 0.0, "rec_tokens": 0.0, "commit": 0.0, "codebook": 0.0, "perplex": 0.0, "rec_globals": 0.0, "count": 0}
        ctx = torch.enable_grad() if do_train else torch.no_grad()
        with ctx:
            for tokens_cont, tokens_id, gvec in loader:
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                gvec = gvec.to(device)

                if do_train:
                    opt.zero_grad()

                out = model(tokens_cont, tokens_id, gvec)
                rec = reconstruction_loss(out["x_hat"], tokens_cont)
                aux = out.get("aux", {})
                loss = rec + aux.get("commit", 0.0) + aux.get("codebook", 0.0) + aux.get("rec_globals", 0.0)

                if do_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.phase1.trainer.grad_clip)
                    opt.step()

                b = tokens_cont.size(0)
                tot["count"] += b
                for k in ("commit", "codebook", "rec_globals"):
                    v = float(aux.get(k, 0.0))
                    tot[k] += v * b
                tot["loss"] += float(loss.item()) * b
                tot["rec_tokens"] += float(rec.item()) * b
                tot["perplex"] += float(aux.get("perplex", 0.0)) * b
        n = max(1, tot["count"])
        for k in list(tot.keys()):
            if k == "count":
                continue
            tot[k] = tot[k] / n
        return tot

    hist_tr_loss = []
    hist_va_loss = []
    hist_rec_tokens = []
    hist_rec_globals = []
    hist_codebook = []
    hist_perplex = []
    hist_epoch_time_s = []
    hist_throughput = []

    best_val = float("inf")
    total_t0 = time.perf_counter()
    progress = TrainingProgressShower(total_epochs=int(cfg.phase1.trainer.epochs), bar_width=30)
    for ep in range(cfg.phase1.trainer.epochs):
        t0 = time.perf_counter()
        tr = run_epoch(train_dl, True)
        va = run_epoch(val_dl, False)
        dt = time.perf_counter() - t0

        # histories
        hist_tr_loss.append(float(tr["loss"]))
        hist_va_loss.append(float(va["loss"]))
        hist_rec_tokens.append(float(va["rec_tokens"]))
        hist_rec_globals.append(float(va.get("rec_globals", 0.0)))
        hist_codebook.append(float(va.get("codebook", 0.0)))
        hist_perplex.append(float(va.get("perplex", 0.0)))
        hist_epoch_time_s.append(float(dt))
        thr = float(tr["count"]) / float(dt) if dt > 0 else 0.0
        hist_throughput.append(thr)

        # event
        payload = build_event_payload(
            moment="on_epoch_end",
            run_dir=outdir,
            epoch=ep,
            split="val",
            train_loss=tr["loss"],
            val_loss=va["loss"],
            metrics={"perplex": va.get("perplex", 0.0)},
            epoch_time_s=dt,
            throughput=thr,
            histories={
                "train_loss": hist_tr_loss,
                "val_loss": hist_va_loss,
                "rec_tokens": hist_rec_tokens,
                "rec_globals": hist_rec_globals,
                "codebook": hist_codebook,
                "perplex": hist_perplex,
                "epoch_time_s": hist_epoch_time_s,
                "throughput": hist_throughput,
            },
            cfg=cfg,
        )
        if outdir and cfg.logging.save_artifacts:
            append_jsonl_event(str(outdir), payload)
            append_scalars_csv(str(outdir), epoch=ep, split="train", train_loss=tr["loss"], val_loss=None, metrics={"perplex": float(tr.get("perplex", 0.0))}, epoch_time_s=dt, throughput=thr, max_memory_mib=None)
            append_scalars_csv(str(outdir), epoch=ep, split="val", train_loss=None, val_loss=va["loss"], metrics={"perplex": float(va.get("perplex", 0.0))}, epoch_time_s=dt, throughput=thr, max_memory_mib=None)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_epoch_end", payload)

        # progress bar with ETA and losses
        progress.update(
            ep,
            dt,
            train_loss=float(tr["loss"]),
            val_loss=float(va["loss"]),
            extras={"perplex": float(va.get("perplex", 0.0))},
        )

        if va["loss"] < best_val and outdir and cfg.logging.save_artifacts:
            best_val = va["loss"]
            torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))

    te = run_epoch(test_dl, False)

    if outdir and cfg.logging.save_artifacts:
        total_time = time.perf_counter() - total_t0
        payload_end = build_event_payload(
            moment="on_train_end",
            run_dir=outdir,
            epoch=int(len(hist_tr_loss) - 1) if hist_tr_loss else -1,
            train_loss=hist_tr_loss[-1] if hist_tr_loss else None,
            val_loss=hist_va_loss[-1] if hist_va_loss else None,
            metrics={"perplex": hist_perplex[-1]} if hist_perplex else {},
            total_time_s=total_time,
            histories={
                "train_loss": hist_tr_loss,
                "val_loss": hist_va_loss,
                "rec_tokens": hist_rec_tokens,
                "rec_globals": hist_rec_globals,
                "codebook": hist_codebook,
                "perplex": hist_perplex,
                "epoch_time_s": hist_epoch_time_s,
                "throughput": hist_throughput,
            },
            cfg=cfg,
        )
        append_jsonl_event(str(outdir), payload_end)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_train_end", payload_end)

        payload_test = build_event_payload(
            moment="on_test_end",
            run_dir=outdir,
            split="test",
            metrics={"test_loss": te["loss"]},
            cfg=cfg,
        )
        append_jsonl_event(str(outdir), payload_test)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_test_end", payload_test)

    return {"best_val_loss": best_val, "test_loss": te["loss"]}
