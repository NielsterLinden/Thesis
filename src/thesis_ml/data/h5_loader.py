import h5py
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, Dataset, TensorDataset


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


class H5ClassificationDataset(Dataset):
    """H5 dataset for classification with label filtering and masking."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = int(cfg.data.n_tokens)  # 18

        # Get selected labels from config (default: [1, 2] for binary)
        selected_labels = cfg.data.classifier.get("selected_labels", [1, 2])
        if isinstance(selected_labels, int | float):
            selected_labels = [int(selected_labels)]
        self.selected_labels = sorted([int(x) for x in selected_labels])
        self.n_classes = len(self.selected_labels)

        # Create label mapping: {original_label: 0_indexed_label}
        self.label_map = {orig: idx for idx, orig in enumerate(self.selected_labels)}

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

        # Create dataset that returns (tokens_cont, ids, globals, mask, label)
        # We'll use a custom dataset class for this
        class ClassificationSplitDataset(Dataset):
            def __init__(self, X, Y, T, mu, sd, create_mask_fn):
                self.X = X
                self.Y = Y
                self.T = T
                self.mu = mu
                self.sd = sd
                self.create_mask = create_mask_fn

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                row = self.X[idx]
                label = self.Y[idx]

                ids = row[: self.T].to(torch.int64)
                globals_ = row[self.T : self.T + 2]
                cont = row[self.T + 2 :].view(self.T, 4)

                tokens_cont = (cont - self.mu[0, 0]) / self.sd[0, 0]
                mask = self.create_mask(ids)

                return tokens_cont, ids, globals_, mask, label

        return ClassificationSplitDataset(X, Y, self.T, self.mu, self.sd, self._create_mask)


class H5BinnedClassificationDataset(Dataset):
    """H5 dataset for binned integer tokens (Ambre's pre-tokenized data).

    Format: [18 integer tokens, 2 integer globals] = 20 integers per event
    Values: 0-885 (0 = padding/no particle)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = int(cfg.data.n_tokens)  # 18

        # Get selected labels
        selected_labels = cfg.data.classifier.get("selected_labels", [1, 2])
        if isinstance(selected_labels, int | float):
            selected_labels = [int(selected_labels)]
        self.selected_labels = sorted([int(x) for x in selected_labels])
        self.n_classes = len(self.selected_labels)
        self.label_map = {orig: idx for idx, orig in enumerate(self.selected_labels)}

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

        class BinnedClassificationSplitDataset(Dataset):
            def __init__(self, X, Y, T, create_mask_fn):
                self.X = X  # [N, 20] integer tokens
                self.Y = Y
                self.T = T
                self.create_mask = create_mask_fn

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                row = self.X[idx]  # [20] integers
                label = self.Y[idx]

                # Split: [18 tokens, 2 globals]
                integer_tokens = row[: self.T].to(torch.long)  # [18]
                globals_ints = row[self.T : self.T + 2].to(torch.long)  # [2]

                # Create attention mask
                mask = self.create_mask(integer_tokens)

                # Return: (integer_tokens, globals_ints, mask, label)
                # Note: For binned tokens, we don't need separate tokens_cont/tokens_id
                return integer_tokens, globals_ints, mask, label

        return BinnedClassificationSplitDataset(X, Y, self.T, self._create_mask)


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
