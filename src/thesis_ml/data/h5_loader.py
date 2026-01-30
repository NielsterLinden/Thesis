import ast

import h5py
import torch
from hydra.utils import to_absolute_path
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset, TensorDataset


def _token_order_permutation(
    T: int,
    shuffle_tokens: bool,
    sort_tokens_by: str | None,
    seed: int,
    idx: int,
    pt_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute permutation of token indices for ordering (Pt sort or shuffle).

    Parameters
    ----------
    T : int
        Sequence length (number of tokens).
    shuffle_tokens : bool
        If True, random permutation (reproducible per sample via seed + idx).
    sort_tokens_by : str | None
        If "pt", sort by Pt (column index 1); requires pt_values. Ignored if shuffle_tokens is True.
    seed : int
        Global seed for reproducibility.
    idx : int
        Sample index (for per-sample shuffle seed).
    pt_values : torch.Tensor | None
        [T] Pt values per token (continuous feature index 1). Required when sort_tokens_by == "pt".

    Returns
    -------
    torch.Tensor
        Indices of shape [T] to apply to token dimension: tokens[perm].
    """
    if shuffle_tokens:
        gen = torch.Generator().manual_seed(seed + idx)
        return torch.randperm(T, generator=gen)
    if sort_tokens_by == "pt" and pt_values is not None:
        # Pt descending = high Pt first (position 0 = leading).
        return torch.argsort(pt_values, descending=True)
    return torch.arange(T, dtype=torch.long)


class H5TokenDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # Resolve to absolute path (Hydra-aware) and open
        with h5py.File(to_absolute_path(str(cfg.data.path)), "r") as f:
            Xtr = f[cfg.data.datasets.x_train][:]
            Xva = f[cfg.data.datasets.x_val][:]  # shapes: [N, 92]
            Xte = f[cfg.data.datasets.x_test][:]

        # Limit samples if specified (for faster testing)
        limit = int(getattr(cfg.data, "limit_samples", 0) or 0)
        if limit > 0:
            Xtr = Xtr[:limit]
            Xva = Xva[: min(limit, len(Xva))]
            Xte = Xte[: min(limit, len(Xte))]

        self.splits = {
            "train": torch.tensor(Xtr, dtype=torch.float32),
            "val": torch.tensor(Xva, dtype=torch.float32),
            "test": torch.tensor(Xte, dtype=torch.float32),
        }

        # constants
        self.T = int(cfg.data.n_tokens)  # 18

        # compute normalization stats for continuous features from TRAIN only
        # layout: [ 0..T-1 ids | T..T+1 globals | T+2.. end continuous (T*4) ]
        train_cont = self.splits["train"][:, self.T + 2 :].view(-1, self.T, 4)
        self.mu = train_cont.mean(dim=(0, 1), keepdim=True)  # [1,1,4]
        self.sd = train_cont.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

        # derive number of types from all splits (assumes non-negative and 0-based)
        def all_ids_min_max():
            min_v = None
            max_v = 0
            for key in ("train", "val", "test"):
                X = self.splits[key]
                ids = X[:, : self.T].to(torch.int64)
                cur_min = int(ids.min().item())
                cur_max = int(ids.max().item())
                min_v = cur_min if min_v is None else min(min_v, cur_min)
                max_v = max(max_v, cur_max)
            return min_v, max_v

        min_id, max_id = all_ids_min_max()
        assert min_id >= 0, "Found negative IDs; remapping required."
        self.num_types = int(max_id) + 1

    def _split_event(self, row: torch.Tensor):
        # row: [92]
        ids = row[: self.T]  # [T]
        globals_ = row[self.T : self.T + 2]  # [2]
        cont = row[self.T + 2 :]  # [T*4]
        tokens_cont = cont.view(self.T, 4)

        # normalize continuous features
        tokens_cont = (tokens_cont - self.mu[0, 0]) / self.sd[0, 0]

        tokens_id = ids.to(torch.int64)
        return tokens_cont, tokens_id, globals_

    def get_split(self, name):
        X = self.splits[name]  # [N, T + 2 + T*4]
        # Vectorized slicing
        ids = X[:, : self.T].to(torch.int64)  # [N, T]
        gl = X[:, self.T : self.T + 2]  # [N, 2]
        cont = X[:, self.T + 2 :].view(-1, self.T, 4)  # [N, T, 4]
        # Normalize continuous features
        tc = (cont - self.mu[0, 0]) / self.sd[0, 0]  # [N, T, 4]
        return TensorDataset(tc, ids, gl)


def make_dataloaders(cfg):
    ds = H5TokenDataset(cfg)
    tr = ds.get_split("train")
    va = ds.get_split("val")
    te = ds.get_split("test")
    return (
        DataLoader(tr, batch_size=cfg.phase1.trainer.batch_size, shuffle=True, num_workers=cfg.data.num_workers),
        DataLoader(va, batch_size=cfg.phase1.trainer.batch_size, shuffle=False, num_workers=cfg.data.num_workers),
        DataLoader(te, batch_size=cfg.phase1.trainer.batch_size, shuffle=False, num_workers=cfg.data.num_workers),
        {
            "n_tokens": cfg.data.n_tokens,
            "cont_dim": 4,
            "globals": cfg.data.globals.size,
            "num_types": ds.num_types,
        },
    )


class ClassificationSplitDataset(Dataset):
    """Dataset for classification splits (raw format)."""

    def __init__(
        self,
        X,
        Y,
        T,
        mu,
        sd,
        pad_id: int = 0,
        *,
        shuffle_tokens: bool = False,
        sort_tokens_by: str | None = None,
        seed: int = 42,
    ):
        self.X = X
        self.Y = Y
        self.T = T
        self.mu_norm = mu[0, 0] if isinstance(mu, torch.Tensor) else mu
        self.sd_norm = sd[0, 0] if isinstance(sd, torch.Tensor) else sd
        self.pad_id = pad_id
        self.shuffle_tokens = shuffle_tokens
        self.sort_tokens_by = sort_tokens_by
        self.seed = seed

    def _create_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from token IDs (nonzero = valid token)."""
        return ids != self.pad_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X[idx]
        label = self.Y[idx]

        ids = row[: self.T].to(torch.int64)
        globals_ = row[self.T : self.T + 2]
        cont = row[self.T + 2 :].view(self.T, 4)

        tokens_cont = (cont - self.mu_norm) / self.sd_norm

        if self.shuffle_tokens or self.sort_tokens_by == "pt":
            pt_vals = tokens_cont[:, 1] if self.sort_tokens_by == "pt" else None
            perm = _token_order_permutation(
                self.T,
                self.shuffle_tokens,
                self.sort_tokens_by,
                self.seed,
                idx,
                pt_values=pt_vals,
            )
            tokens_cont = tokens_cont[perm]
            ids = ids[perm]

        mask = self._create_mask(ids)
        return tokens_cont, ids, globals_, mask, label


class BinnedClassificationSplitDataset(Dataset):
    """Dataset for classification splits (binned format)."""

    def __init__(self, X, Y, T, pad_id: int = 0, *, shuffle_tokens: bool = False, seed: int = 42):
        self.X = X  # [N, 20] integer tokens
        self.Y = Y
        self.T = T
        self.pad_id = pad_id
        self.shuffle_tokens = shuffle_tokens
        self.seed = seed

    def _create_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create attention mask from tokens (nonzero = valid token)."""
        return tokens != self.pad_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X[idx]  # [20] integers
        label = self.Y[idx]

        integer_tokens = row[: self.T].to(torch.long)  # [18]
        globals_ints = row[self.T : self.T + 2].to(torch.long)  # [2]

        if self.shuffle_tokens:
            perm = _token_order_permutation(self.T, True, None, self.seed, idx, pt_values=None)
            integer_tokens = integer_tokens[perm]

        mask = self._create_mask(integer_tokens)
        return integer_tokens, globals_ints, mask, label


def _parse_selected_labels(selected_labels):
    """Parse selected_labels from config, handling various input formats.

    Supports:
    - List: [1, 2]
    - ListConfig (OmegaConf): ListConfig(['1,2']) or ListConfig([1, 2])
    - String representation of list: "[1,2]" or '"[1,2]"'
    - Comma-separated string: "1,2" (for Hydra sweeper compatibility)
    - Single int/float: 1 or 2.0

    Parameters
    ----------
    selected_labels : Any
        Input from config (list, ListConfig, string, int, float)

    Returns
    -------
    list[int]
        Parsed list of integer labels
    """
    # Handle OmegaConf ListConfig (from Hydra)
    if isinstance(selected_labels, ListConfig):
        # Convert to Python list and process
        # If ListConfig contains strings like ['1,2'], parse the first element
        # If ListConfig contains integers like [1, 2], use directly
        if len(selected_labels) == 0:
            raise ValueError("selected_labels ListConfig is empty")

        # If it's a single-element list with a comma-separated string, parse it
        if len(selected_labels) == 1 and isinstance(selected_labels[0], str) and "," in str(selected_labels[0]):
            # This is likely from Hydra sweeper: ListConfig(['1,2'])
            return _parse_selected_labels(selected_labels[0])
        else:
            # ListConfig with multiple elements or integers: ListConfig([1, 2]) or ListConfig(['1', '2'])
            # Convert to list and parse each element
            return sorted([int(x) for x in selected_labels])

    # Handle single int/float
    if isinstance(selected_labels, int | float):
        return [int(selected_labels)]

    # Handle string (from Hydra sweeper or config)
    if isinstance(selected_labels, str):
        s = selected_labels.strip()

        # Remove outer quotes if present (e.g., '"[1,2]"' -> "[1,2]")
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]

        # Check if it's a comma-separated list of numbers (e.g., "1,2") or colon-separated (e.g., "1:2")
        # Support both formats for flexibility
        if ("," in s or ":" in s) and not s.startswith("[") and not s.startswith("("):
            # Use colon as separator if present, otherwise comma
            separator = ":" if ":" in s else ","
            # Parse separated values
            try:
                return sorted([int(x.strip()) for x in s.split(separator)])
            except ValueError as e:
                raise ValueError(f"Cannot parse {separator}-separated selected_labels '{selected_labels}': {e}") from e

        # Try to parse as Python literal (list, tuple, etc.)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list | tuple):
                return [int(x) for x in parsed]
            elif isinstance(parsed, int | float):
                return [int(parsed)]
            else:
                raise ValueError(f"Cannot parse selected_labels string '{selected_labels}': parsed to {type(parsed)}")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Cannot parse selected_labels string '{selected_labels}': {e}") from e

    # Handle list/tuple
    if isinstance(selected_labels, list | tuple):
        return sorted([int(x) for x in selected_labels])

    raise TypeError(f"selected_labels must be list, tuple, ListConfig, int, float, or string, got {type(selected_labels)}")


def _normalize_label_groups(cfg) -> tuple[list[dict], dict[int, int], list[int]]:
    """Normalize label configuration to label_groups format.

    Converts any of the three input formats (label_groups, signal_vs_background, selected_labels)
    into a unified label_groups structure. This is the single canonical mechanism for label mapping.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with data.classifier.* keys

    Returns
    -------
    tuple[list[dict], dict[int, int], list[int]]
        - label_groups: List of {name: str, labels: list[int]} dicts
        - label_map: Dict mapping original label -> class index (0-indexed)
        - selected_labels: Union of all labels from all groups (for data filtering)
    """
    classifier_cfg = cfg.data.classifier

    # Priority 1: Use label_groups if explicitly provided
    if "label_groups" in classifier_cfg and classifier_cfg.label_groups is not None:
        label_groups_raw = classifier_cfg.label_groups
        label_groups = []
        label_map = {}
        selected_labels_set = set()

        for class_idx, group in enumerate(label_groups_raw):
            name = group.get("name", f"Class-{class_idx}")
            labels_raw = group.get("labels", [])
            labels = sorted(_parse_selected_labels(labels_raw))
            label_groups.append({"name": name, "labels": labels})
            # Map each label in this group to the class index
            for label in labels:
                if label in label_map:
                    raise ValueError(f"Label {label} appears in multiple groups")
                label_map[label] = class_idx
                selected_labels_set.add(label)

        selected_labels = sorted(selected_labels_set)
        return label_groups, label_map, selected_labels

    # Priority 2: Convert signal_vs_background to binary label_groups
    signal_vs_bg = classifier_cfg.get("signal_vs_background", None)
    if signal_vs_bg is not None:
        signal_label = int(signal_vs_bg.get("signal"))
        bg_labels_raw = signal_vs_bg.get("background", [])
        background_labels = sorted(_parse_selected_labels(bg_labels_raw))

        # Ensure signal label is not in background labels
        if signal_label in background_labels:
            raise ValueError(f"Signal label {signal_label} cannot be in background labels {background_labels}")

        # Create binary label_groups: background (class 0) → signal (class 1)
        label_groups = [
            {"name": "background", "labels": background_labels},
            {"name": "signal", "labels": [signal_label]},
        ]
        label_map = {signal_label: 1}
        for bg_label in background_labels:
            label_map[bg_label] = 0
        selected_labels = sorted([signal_label] + background_labels)
        return label_groups, label_map, selected_labels

    # Priority 3: Convert selected_labels to one group per label
    selected_labels_raw = classifier_cfg.get("selected_labels", [1, 2])
    selected_labels = sorted(_parse_selected_labels(selected_labels_raw))

    # Default ProcessID to name mapping
    process_id_names = {1: "4t", 2: "ttH", 3: "ttW", 4: "ttWW", 5: "ttZ"}

    label_groups = []
    label_map = {}
    for class_idx, label in enumerate(selected_labels):
        name = process_id_names.get(label, f"Class-{label}")
        label_groups.append({"name": name, "labels": [label]})
        label_map[label] = class_idx

    return label_groups, label_map, selected_labels


class H5ClassificationDataset(Dataset):
    """H5 dataset for classification with label filtering and masking."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = int(cfg.data.n_tokens)  # 18

        # Normalize label configuration to unified label_groups format
        label_groups, label_map, selected_labels = _normalize_label_groups(cfg)
        self.label_groups = label_groups
        self.label_map = label_map
        self.selected_labels = selected_labels
        self.n_classes = len(label_groups)

        # Load data and labels
        with h5py.File(to_absolute_path(str(cfg.data.path)), "r") as f:
            Xtr = f[cfg.data.datasets.x_train][:]
            Xva = f[cfg.data.datasets.x_val][:]
            Xte = f[cfg.data.datasets.x_test][:]

            # Load labels (note: HDF5 file has inconsistent casing: Y_train but y_val and y_test)
            # Use config values to handle the misnomer
            y_train_key = cfg.data.datasets.get("y_train", "Y_train")
            y_val_key = cfg.data.datasets.get("y_val", "y_val")
            y_test_key = cfg.data.datasets.get("y_test", "y_test")

            Ytr = f.get(y_train_key, None)
            Yva = f.get(y_val_key, None)
            Yte = f.get(y_test_key, None)

            if Ytr is None:
                raise ValueError(f"{y_train_key} not found in H5 file")
            Ytr = Ytr[:]
            Yva = Yva[:] if Yva is not None else Ytr[: len(Xva)]  # Fallback if missing
            Yte = Yte[:] if Yte is not None else Ytr[: len(Xte)]  # Fallback if missing

        # Limit samples if specified
        limit = int(getattr(cfg.data, "limit_samples", 0) or 0)
        if limit > 0:
            Xtr = Xtr[:limit]
            Xva = Xva[: min(limit, len(Xva))]
            Xte = Xte[: min(limit, len(Xte))]
            Ytr = Ytr[:limit]
            Yva = Yva[: min(limit, len(Yva))]
            Yte = Yte[: min(limit, len(Yte))]

        # Filter events to only selected labels
        def filter_and_map(X, Y):

            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            selected_tensor = torch.tensor(self.selected_labels, dtype=torch.float32)
            mask = torch.isin(Y_tensor, selected_tensor)
            mask_np = mask.numpy()
            X_filtered = X[mask_np]
            Y_filtered = Y[mask_np]
            # Map labels to 0-indexed
            Y_mapped = torch.tensor([self.label_map[int(y)] for y in Y_filtered], dtype=torch.long)
            return X_filtered, Y_mapped

        Xtr, Ytr = filter_and_map(Xtr, Ytr)
        Xva, Yva = filter_and_map(Xva, Yva)
        Xte, Yte = filter_and_map(Xte, Yte)

        self.splits = {
            "train": (torch.tensor(Xtr, dtype=torch.float32), Ytr),
            "val": (torch.tensor(Xva, dtype=torch.float32), Yva),
            "test": (torch.tensor(Xte, dtype=torch.float32), Yte),
        }

        # Compute normalization stats from TRAIN only
        train_cont = self.splits["train"][0][:, self.T + 2 :].view(-1, self.T, 4)
        self.mu = train_cont.mean(dim=(0, 1), keepdim=True)
        self.sd = train_cont.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

        # Derive number of types from all splits (assumes non-negative and 0-based)
        def all_ids_min_max():
            min_v = None
            max_v = 0
            for key in ("train", "val", "test"):
                X = self.splits[key][0]
                ids = X[:, : self.T].to(torch.int64)
                cur_min = int(ids.min().item())
                cur_max = int(ids.max().item())
                min_v = cur_min if min_v is None else min(min_v, cur_min)
                max_v = max(max_v, cur_max)
            return min_v, max_v

        min_id, max_id = all_ids_min_max()
        assert min_id >= 0, "Found negative IDs; remapping required."
        self.num_types = int(max_id) + 1

    def _create_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from token IDs (nonzero = valid token)."""
        # Mask is True for valid tokens (ID != 0), False for padding
        return ids != 0

    def __len__(self):
        return len(self.splits["train"][0])

    def __getitem__(self, idx):
        # For now, return train split (will be handled by get_split)
        X, Y = self.splits["train"]
        row = X[idx]
        label = Y[idx]

        # Split event
        ids = row[: self.T].to(torch.int64)
        globals_ = row[self.T : self.T + 2]
        cont = row[self.T + 2 :].view(self.T, 4)

        # Normalize continuous features
        tokens_cont = (cont - self.mu[0, 0]) / self.sd[0, 0]

        # Create attention mask
        mask = self._create_mask(ids)

        return tokens_cont, ids, globals_, mask, label

    def get_split(self, name):
        X, Y = self.splits[name]
        shuffle_tokens = bool(getattr(self.cfg.data, "shuffle_tokens", False))
        sort_tokens_by = getattr(self.cfg.data, "sort_tokens_by", None)
        try:
            seed = int(self.cfg.classifier.trainer.seed)
        except (AttributeError, TypeError):
            try:
                seed = int(self.cfg.trainer.seed)
            except (AttributeError, TypeError):
                seed = int(getattr(self.cfg, "seed", 42))
        return ClassificationSplitDataset(
            X,
            Y,
            self.T,
            self.mu,
            self.sd,
            pad_id=0,
            shuffle_tokens=shuffle_tokens,
            sort_tokens_by=sort_tokens_by,
            seed=seed,
        )


class H5BinnedClassificationDataset(Dataset):
    """H5 dataset for binned integer tokens (Ambre's pre-tokenized data).

    Format: [18 integer tokens, 2 integer globals] = 20 integers per event
    Values: 0-885 (0 = padding/no particle)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = int(cfg.data.n_tokens)  # 18

        # Normalize label configuration to unified label_groups format
        label_groups, label_map, selected_labels = _normalize_label_groups(cfg)
        self.label_groups = label_groups
        self.label_map = label_map
        self.selected_labels = selected_labels
        self.n_classes = len(label_groups)

        # Load binned token data
        with h5py.File(to_absolute_path(str(cfg.data.path)), "r") as f:
            Xtr = f[cfg.data.datasets.x_train][:]  # [N, 20] integers
            Xva = f[cfg.data.datasets.x_val][:]
            Xte = f[cfg.data.datasets.x_test][:]

            # Load labels
            y_train_key = cfg.data.datasets.get("y_train", "y_train")
            y_val_key = cfg.data.datasets.get("y_val", "y_val")
            y_test_key = cfg.data.datasets.get("y_test", "y_test")

            Ytr = f.get(y_train_key, None)
            Yva = f.get(y_val_key, None)
            Yte = f.get(y_test_key, None)

            if Ytr is None:
                raise ValueError(f"{y_train_key} not found in H5 file")
            Ytr = Ytr[:]
            Yva = Yva[:] if Yva is not None else Ytr[: len(Xva)]
            Yte = Yte[:] if Yte is not None else Ytr[: len(Xte)]

        # Limit samples if specified
        limit = int(getattr(cfg.data, "limit_samples", 0) or 0)
        if limit > 0:
            Xtr = Xtr[:limit]
            Xva = Xva[: min(limit, len(Xva))]
            Xte = Xte[: min(limit, len(Xte))]
            Ytr = Ytr[:limit]
            Yva = Yva[: min(limit, len(Yva))]
            Yte = Yte[: min(limit, len(Yte))]

        # Filter events to only selected labels
        def filter_and_map(X, Y):

            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            selected_tensor = torch.tensor(self.selected_labels, dtype=torch.float32)
            mask = torch.isin(Y_tensor, selected_tensor)
            mask_np = mask.numpy()
            X_filtered = X[mask_np]
            Y_filtered = Y[mask_np]
            Y_mapped = torch.tensor([self.label_map[int(y)] for y in Y_filtered], dtype=torch.long)
            return X_filtered, Y_mapped

        Xtr, Ytr = filter_and_map(Xtr, Ytr)
        Xva, Yva = filter_and_map(Xva, Yva)
        Xte, Yte = filter_and_map(Xte, Yte)

        self.splits = {
            "train": (torch.tensor(Xtr, dtype=torch.long), Ytr),
            "val": (torch.tensor(Xva, dtype=torch.long), Yva),
            "test": (torch.tensor(Xte, dtype=torch.long), Yte),
        }

        # Determine vocab size from data (max token value + 1)
        all_tokens = torch.cat(
            [
                self.splits["train"][0],
                self.splits["val"][0],
                self.splits["test"][0],
            ]
        )
        self.vocab_size = int(all_tokens.max().item()) + 1  # 0-885 → 886

    def _create_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create attention mask from tokens (nonzero = valid token)."""
        # Mask is True for valid tokens (token != 0), False for padding
        return tokens != 0

    def get_split(self, name):
        X, Y = self.splits[name]
        shuffle_tokens = bool(getattr(self.cfg.data, "shuffle_tokens", False))
        try:
            seed = int(self.cfg.classifier.trainer.seed)
        except (AttributeError, TypeError):
            try:
                seed = int(self.cfg.trainer.seed)
            except (AttributeError, TypeError):
                seed = int(getattr(self.cfg, "seed", 42))
        return BinnedClassificationSplitDataset(X, Y, self.T, pad_id=0, shuffle_tokens=shuffle_tokens, seed=seed)


def make_classification_dataloaders(cfg):
    """Create dataloaders for classification with labels and masks.

    Supports both raw H5 tokens and binned integer tokens.
    Detects format based on data path or config flag.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with data.* and data.classifier.selected_labels keys
        If cfg.data.use_binned_tokens is True, uses binned format

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, dict]
        Train, val, test dataloaders and metadata dict
    """
    # Check if we should use binned tokens
    use_binned = cfg.data.get("use_binned_tokens", False)

    if use_binned:
        ds = H5BinnedClassificationDataset(cfg)
        meta_vocab_size = ds.vocab_size
    else:
        ds = H5ClassificationDataset(cfg)
        meta_vocab_size = None

    tr = ds.get_split("train")
    va = ds.get_split("val")
    te = ds.get_split("test")

    batch_size = cfg.classifier.trainer.get("batch_size", cfg.data.get("batch_size", 32))
    num_workers = cfg.data.get("num_workers", 4)

    meta = {
        "n_tokens": cfg.data.n_tokens,
        "has_globals": cfg.data.globals.get("present", False),
        "n_classes": ds.n_classes,
    }

    if use_binned:
        meta["vocab_size"] = meta_vocab_size
        meta["token_feat_dim"] = None  # Not applicable for binned
        meta["num_types"] = None  # Not applicable for binned
    else:
        meta["token_feat_dim"] = 4  # Continuous features per token
        meta["vocab_size"] = None
        meta["num_types"] = ds.num_types  # For identity tokenizer

    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        meta,
    )
