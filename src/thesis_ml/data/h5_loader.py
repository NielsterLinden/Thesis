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
