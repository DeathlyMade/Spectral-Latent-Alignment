"""
benchmark_timing.py
====================
Benchmarks the wall-clock runtime of every model implemented in this repo:
  - SLA   (Spectral Latent Alignment - our method)
  - CMF   (Collective Matrix Factorisation)
  - I-DARec  (Item-based DARec)
  - U-DARec  (User-based DARec)
  - DARec    (original TF1 DARec)

For each model × dataset pair it records:
  - elapsed wall-clock time  (seconds)
  - RMSE on the target domain test split

Results are printed to stdout in a formatted table AND saved to
  timing_results.txt   (plain text table)
  timing_results.csv   (machine-readable)

Usage
-----
  python benchmark_timing.py [--models SLA CMF IDAREC UDAREC DAREC]
                             [--datasets 0 1 2 3]
                             [--sla_epochs 50]
                             [--cmf_epochs 30]
                             [--darec_epochs 1500]
                             [--idarec_epochs 70]
                             [--udarec_epochs 20]
                             [--base_path /path/to/data/dir]
                             [--output_dir .]

Example (quick smoke-test, 1 dataset, SLA + CMF only):
  python benchmark_timing.py --models SLA CMF --datasets 0 --sla_epochs 5 --cmf_epochs 5
"""

import argparse
import os
import sys
import time
import warnings
import csv
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Runtime + RMSE benchmark for CDR models")

parser.add_argument('--models', nargs='+',
                    default=['SLA', 'CMF', 'IDAREC', 'UDAREC', 'DAREC'],
                    choices=['SLA', 'CMF', 'IDAREC', 'UDAREC', 'DAREC'],
                    help="Which models to benchmark")

parser.add_argument('--datasets', nargs='+', type=int, default=[0, 1, 2, 3],
                    choices=[0, 1, 2, 3],
                    help="Dataset pair indices to run (0-3)")

parser.add_argument('--base_path', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Directory that contains the .npy pre-processed files")

parser.add_argument('--output_dir', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Directory to write timing_results.txt / .csv")

# Per-model epoch counts (match the defaults used in the original scripts)
parser.add_argument('--sla_epochs',    type=int, default=50)
parser.add_argument('--cmf_epochs',    type=int, default=30)
parser.add_argument('--darec_epochs',  type=int, default=1500)
parser.add_argument('--idarec_epochs', type=int, default=70)
parser.add_argument('--udarec_epochs', type=int, default=20)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_PAIRS = [
    ("Office Products → Movies and TV",
     "ratings_Office_Products.csv",
     "ratings_Movies_and_TV.csv"),
    ("Sports and Outdoors → CDs and Vinyls",
     "ratings_Sports_and_Outdoors.csv",
     "ratings_CDs_and_Vinyl.csv"),
    ("Android Apps → Video Games",
     "ratings_Apps_for_Android.csv",
     "ratings_Video_Games.csv"),
    ("Toys and Games → Automotive",
     "ratings_Toys_and_Games.csv",
     "ratings_Automotive.csv"),
]

BASE = args.base_path

# ---------------------------------------------------------------------------
# Helper: load a dataset pair
# ---------------------------------------------------------------------------
def load_dataset(s_file, t_file):
    """Load pre-processed .npy files through Data_Preprocessing.Mydata."""
    # Add repo root to path so local modules are importable
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from Data_Preprocessing import Mydata

    s_path = os.path.join(BASE, s_file)
    t_path = os.path.join(BASE, t_file)

    try:
        dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
    except Exception:
        print("  [!] Preprocessed .npy not found – generating from CSV (slow first run)…")
        dataset = Mydata(s_path, t_path, train=None, preprocessed=False)

    return dataset.S_data, dataset.T_data, dataset.train_indices, dataset.test_indices


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


# ---------------------------------------------------------------------------
# Model runners – each returns (rmse, elapsed_seconds)
# ---------------------------------------------------------------------------

def run_sla(S_data, T_data, train_indices, test_indices, epochs):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from SLA import train_sla

    with Timer() as t:
        rmse = train_sla(S_data, T_data, train_indices, test_indices,
                         k=100, epochs=epochs)
    return rmse, t.elapsed


def run_cmf(S_data, T_data, train_indices, test_indices, epochs):
    """
    CMF is defined to receive file paths in its public API.
    We replicate the training loop inline here so we can pass numpy arrays
    directly (the same approach used in run_all.py for SLA).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # ---- inline CMF model (mirrors CMF.py) ----
    class CMFModel(nn.Module):
        def __init__(self, num_users, num_items_S, num_items_T, k=50):
            super().__init__()
            self.U    = nn.Embedding(num_users, k)
            self.V_S  = nn.Embedding(num_items_S, k)
            self.V_T  = nn.Embedding(num_items_T, k)
            self.b_U  = nn.Embedding(num_users, 1)
            self.b_S  = nn.Embedding(num_items_S, 1)
            self.b_T  = nn.Embedding(num_items_T, 1)
            self.global_mean_S = nn.Parameter(torch.zeros(1))
            self.global_mean_T = nn.Parameter(torch.zeros(1))
            nn.init.normal_(self.U.weight,   std=0.01)
            nn.init.normal_(self.V_S.weight, std=0.01)
            nn.init.normal_(self.V_T.weight, std=0.01)

        def forward_S(self, u, i):
            return (self.global_mean_S
                    + self.b_U(u).squeeze()
                    + self.b_S(i).squeeze()
                    + (self.U(u) * self.V_S(i)).sum(1))

        def forward_T(self, u, i):
            return (self.global_mean_T
                    + self.b_U(u).squeeze()
                    + self.b_T(i).squeeze()
                    + (self.U(u) * self.V_T(i)).sum(1))

    with Timer() as t:
        num_users    = S_data.shape[0]
        num_items_S  = S_data.shape[1]
        num_items_T  = T_data.shape[1]

        # Build source loader (all rows)
        u_S, i_S = np.where(S_data > 0)
        r_S = S_data[u_S, i_S].astype(np.float32)

        # Build target loader (train rows only)
        T_train = np.zeros_like(T_data)
        T_train[train_indices] = T_data[train_indices]
        u_T, i_T = np.where(T_train > 0)
        r_T = T_data[u_T, i_T].astype(np.float32)

        ds_S = TensorDataset(torch.tensor(u_S, dtype=torch.long),
                             torch.tensor(i_S, dtype=torch.long),
                             torch.tensor(r_S))
        ds_T = TensorDataset(torch.tensor(u_T, dtype=torch.long),
                             torch.tensor(i_T, dtype=torch.long),
                             torch.tensor(r_T))
        ld_S = DataLoader(ds_S, batch_size=512, shuffle=True)
        ld_T = DataLoader(ds_T, batch_size=512, shuffle=True)

        model = CMFModel(num_users, num_items_S, num_items_T, k=50)
        with torch.no_grad():
            model.global_mean_S.fill_(float(r_S.mean()))
            model.global_mean_T.fill_(float(r_T.mean()))

        opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        crit = nn.MSELoss()

        for _ in range(epochs):
            model.train()
            it_S = iter(ld_S)
            it_T = iter(ld_T)
            for _ in range(max(len(ld_S), len(ld_T))):
                opt.zero_grad()
                try:   b_uS, b_iS, b_rS = next(it_S)
                except StopIteration:
                    it_S = iter(ld_S); b_uS, b_iS, b_rS = next(it_S)
                try:   b_uT, b_iT, b_rT = next(it_T)
                except StopIteration:
                    it_T = iter(ld_T); b_uT, b_iT, b_rT = next(it_T)
                loss = 0.5 * crit(model.forward_S(b_uS, b_iS), b_rS) \
                     + 0.5 * crit(model.forward_T(b_uT, b_iT), b_rT)
                loss.backward()
                opt.step()

        # Evaluate on target test set
        model.eval()
        u_te, i_te = np.where(T_data[test_indices] > 0)
        real_u = test_indices[u_te]
        y_true = T_data[real_u, i_te].astype(np.float32)
        with torch.no_grad():
            preds = model.forward_T(
                torch.tensor(real_u, dtype=torch.long),
                torch.tensor(i_te,   dtype=torch.long)).numpy()
        preds = np.clip(preds, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))

    return rmse, t.elapsed


def run_idarec(S_data, T_data, train_indices, test_indices, epochs):
    """
    Runs the I-DARec pipeline:
      1. Train source AutoRec
      2. Train target AutoRec
      3. Train I-DARec transfer model and evaluate
    """
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import math

    repo_root = os.path.dirname(os.path.abspath(__file__))
    idarec_dir = os.path.join(repo_root, 'I-DARec')
    if idarec_dir not in sys.path:
        sys.path.insert(0, idarec_dir)

    from AutoRec import I_AutoRec
    from I_DArec import I_DArec, MRMSELoss, DArec_Loss
    from torch.utils.data import DataLoader, TensorDataset, Dataset

    # Build a minimal Dataset compatible with the existing code
    class ArrayDataset(Dataset):
        def __init__(self, S, T, indices):
            self.S = torch.tensor(S[indices], dtype=torch.float32)
            self.T = torch.tensor(T[indices], dtype=torch.float32)
            self.Sy = torch.zeros(len(indices), 1)
            self.Ty = torch.ones(len(indices), 1)

        def __len__(self):
            return len(self.S)

        def __getitem__(self, idx):
            return self.S[idx], self.T[idx], self.Sy[idx], self.Ty[idx]

    with Timer() as t:
        train_ds = ArrayDataset(S_data, T_data, train_indices)
        test_ds  = ArrayDataset(S_data, T_data, test_indices)
        train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False)

        # Dimensions – I-DARec is item-based: items as "users" in autoencoder
        n_users_S = S_data.shape[0]   # rows shared
        n_items_S = S_data.shape[1]
        n_items_T = T_data.shape[1]

        RMSE_fn = MRMSELoss()
        crit    = DArec_Loss()

        # ---- 1. Source AutoRec ----
        s_autorec = I_AutoRec(n_users=n_users_S, n_items=n_items_S, n_factors=200)
        opt_sa = optim.Adam(s_autorec.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(50):
            s_autorec.train()
            for d in train_ld:
                src = d[0]
                opt_sa.zero_grad()
                _, pred = s_autorec(src)
                loss, _ = RMSE_fn(pred, src)
                loss.backward()
                opt_sa.step()

        # ---- 2. Target AutoRec ----
        t_autorec = I_AutoRec(n_users=n_users_S, n_items=n_items_T, n_factors=200)
        opt_ta = optim.Adam(t_autorec.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(50):
            t_autorec.train()
            for d in train_ld:
                tgt = d[1]
                opt_ta.zero_grad()
                _, pred = t_autorec(tgt)
                loss, _ = RMSE_fn(pred, tgt)
                loss.backward()
                opt_ta.step()

        # ---- 3. I-DARec ----
        class SimpleArgs:
            n_factors       = 200
            n_users         = n_users_S
            S_n_items       = n_items_S
            T_n_items       = n_items_T
            RPE_hidden_size = 200

        net_args = SimpleArgs()
        net = I_DArec(net_args)
        net.S_autorec.load_state_dict(s_autorec.state_dict())
        net.T_autorec.load_state_dict(t_autorec.state_dict())

        opt = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=1e-3, weight_decay=1e-4)

        for _ in range(epochs):
            net.train()
            for d in train_ld:
                src, tgt, sl, tl = d
                sl = sl.squeeze(1).long()
                tl = tl.squeeze(1).long()
                opt.zero_grad()
                co, sp, tp = net(src, True)
                loss, _, _ = crit(co, sp, tp, src, tgt, sl)
                co2, sp2, tp2 = net(tgt, False)
                loss2, _, _ = crit(co2, sp2, tp2, src, tgt, tl)
                (loss + loss2).backward()
                opt.step()

        # ---- Evaluate ----
        net.eval()
        total_sq = 0.0
        total_n  = 0
        with torch.no_grad():
            for d in test_ld:
                src, tgt, sl, _ = d
                _, _, tp = net(src, True)
                tp_np   = tp.numpy()
                tgt_np  = tgt.numpy()
                mask    = tgt_np > 0
                if mask.sum() == 0:
                    continue
                preds   = np.clip(tp_np[mask], 1.0, 5.0)
                truth   = tgt_np[mask]
                total_sq += float(np.sum((preds - truth) ** 2))
                total_n  += int(mask.sum())

        rmse = float(np.sqrt(total_sq / total_n)) if total_n > 0 else float('nan')

    return rmse, t.elapsed


def run_udarec(S_data, T_data, train_indices, test_indices, epochs):
    """
    Runs the U-DARec pipeline:
      1. Train source AutoRec
      2. Train target AutoRec
      3. Train U-DARec transfer model and evaluate
    """
    import torch
    import torch.optim as optim
    import math

    repo_root = os.path.dirname(os.path.abspath(__file__))
    udarec_dir = os.path.join(repo_root, 'U-DARec')
    if udarec_dir not in sys.path:
        sys.path.insert(0, udarec_dir)

    from AutoRec import U_AutoRec
    from model import U_DArec, MRMSELoss, DArec_Loss
    from torch.utils.data import DataLoader, Dataset

    class ArrayDataset(Dataset):
        def __init__(self, S, T, indices):
            self.S  = torch.tensor(S[indices], dtype=torch.float32)
            self.T  = torch.tensor(T[indices], dtype=torch.float32)
            self.Sy = torch.zeros(len(indices), 1)
            self.Ty = torch.ones(len(indices), 1)

        def __len__(self): return len(self.S)
        def __getitem__(self, idx):
            return self.S[idx], self.T[idx], self.Sy[idx], self.Ty[idx]

    with Timer() as t:
        train_ds = ArrayDataset(S_data, T_data, train_indices)
        test_ds  = ArrayDataset(S_data, T_data, test_indices)
        train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False)

        n_users  = S_data.shape[0]
        n_items_S = S_data.shape[1]
        n_items_T = T_data.shape[1]

        RMSE_fn = MRMSELoss()
        crit    = DArec_Loss()

        # ---- 1. Source AutoRec ----
        s_autorec = U_AutoRec(n_users=n_users, n_items=n_items_S, n_factors=200)
        opt_sa = torch.optim.Adam(s_autorec.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(20):
            s_autorec.train()
            for d in train_ld:
                src = d[0]
                opt_sa.zero_grad()
                _, pred = s_autorec(src)
                loss, _ = RMSE_fn(pred, src)
                loss.backward()
                opt_sa.step()

        # ---- 2. Target AutoRec ----
        t_autorec = U_AutoRec(n_users=n_users, n_items=n_items_T, n_factors=200)
        opt_ta = torch.optim.Adam(t_autorec.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(20):
            t_autorec.train()
            for d in train_ld:
                tgt = d[1]
                opt_ta.zero_grad()
                _, pred = t_autorec(tgt)
                loss, _ = RMSE_fn(pred, tgt)
                loss.backward()
                opt_ta.step()

        # ---- 3. U-DARec ----
        class SimpleArgs:
            n_factors       = 200
            n_users         = n_users
            S_n_items       = n_items_S
            T_n_items       = n_items_T
            RPE_hidden_size = 200

        net_args = SimpleArgs()
        net = U_DArec(net_args)
        net.S_autorec.load_state_dict(s_autorec.state_dict())
        net.T_autorec.load_state_dict(t_autorec.state_dict())

        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=1e-3, weight_decay=1e-5)

        alpha = 1.0
        for _ in range(epochs):
            net.train()
            for d in train_ld:
                src, tgt, sl, tl = d
                sl = sl.squeeze(1).long()
                tl = tl.squeeze(1).long()
                opt.zero_grad()
                co, sp, tp = net(src, alpha, True)
                loss, _, _ = crit(co, sp, tp, src, tgt, sl)
                co2, sp2, tp2 = net(tgt, alpha, False)
                loss2, _, _ = crit(co2, sp2, tp2, src, tgt, tl)
                (loss + loss2).backward()
                opt.step()

        # ---- Evaluate ----
        net.eval()
        total_sq = 0.0
        total_n  = 0
        with torch.no_grad():
            for d in test_ld:
                src, tgt, sl, _ = d
                _, _, tp = net(src, alpha, True)
                tp_np  = tp.numpy()
                tgt_np = tgt.numpy()
                mask   = tgt_np > 0
                if mask.sum() == 0:
                    continue
                preds  = np.clip(tp_np[mask], 1.0, 5.0)
                truth  = tgt_np[mask]
                total_sq += float(np.sum((preds - truth) ** 2))
                total_n  += int(mask.sum())

        rmse = float(np.sqrt(total_sq / total_n)) if total_n > 0 else float('nan')

    return rmse, t.elapsed


def run_darec(S_data, T_data, train_indices, test_indices, epochs):
    """Original TF1 DARec (via run_darec.py)."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from run_darec import run_darec_on_data

    with Timer() as t:
        rmse = run_darec_on_data(S_data, T_data, train_indices, test_indices,
                                 epochs=epochs)
    return rmse, t.elapsed


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------
MODEL_FNS = {
    'SLA':    (run_sla,    'sla_epochs'),
    'CMF':    (run_cmf,    'cmf_epochs'),
    'IDAREC': (run_idarec, 'idarec_epochs'),
    'UDAREC': (run_udarec, 'udarec_epochs'),
    'DAREC':  (run_darec,  'darec_epochs'),
}

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
print("=" * 90)
print(f"CDR Benchmark  –  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Models:   {args.models}")
print(f"Datasets: {[DATASET_PAIRS[i][0] for i in args.datasets]}")
print("=" * 90)

records = []   # list of dicts for CSV

for ds_idx in args.datasets:
    name, s_file, t_file = DATASET_PAIRS[ds_idx]
    print(f"\n{'─'*90}")
    print(f"Dataset pair: {name}")
    print(f"{'─'*90}")

    # Load once, reuse for all models
    print("  Loading data … ", end='', flush=True)
    with Timer() as tload:
        S_data, T_data, train_indices, test_indices = load_dataset(s_file, t_file)
    print(f"done ({fmt_time(tload.elapsed)})  "
          f"S:{S_data.shape}  T:{T_data.shape}  "
          f"train_users:{len(train_indices)}  test_users:{len(test_indices)}")

    for model_key in args.models:
        fn, epoch_attr = MODEL_FNS[model_key]
        n_epochs = getattr(args, epoch_attr)
        print(f"\n  [{model_key}]  epochs={n_epochs}", flush=True)
        try:
            rmse, elapsed = fn(S_data, T_data, train_indices, test_indices, n_epochs)
            status = 'OK'
        except Exception as exc:
            print(f"    ERROR: {exc}")
            rmse, elapsed, status = float('nan'), float('nan'), f'ERROR: {exc}'

        print(f"    ✓  RMSE = {rmse:.4f}   Time = {fmt_time(elapsed)}")
        records.append({
            'dataset':   name,
            'model':     model_key,
            'epochs':    n_epochs,
            'rmse':      rmse,
            'time_s':    elapsed,
            'time_fmt':  fmt_time(elapsed) if not np.isnan(elapsed) else 'N/A',
            'status':    status,
        })

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)

col_w = [42, 10, 10, 12, 10]
headers = ['Dataset', 'Model', 'Epochs', 'RMSE', 'Time']
row_fmt = "  {:<42} {:<10} {:<10} {:<12} {:<10}"
print(row_fmt.format(*headers))
print("  " + "─" * 82)

for r in records:
    rmse_str = f"{r['rmse']:.4f}" if not np.isnan(r['rmse']) else "N/A"
    print(row_fmt.format(
        r['dataset'][:42],
        r['model'],
        str(r['epochs']),
        rmse_str,
        r['time_fmt'],
    ))

# ---------------------------------------------------------------------------
# Save to files
# ---------------------------------------------------------------------------
out_txt = os.path.join(args.output_dir, 'timing_results.txt')
out_csv = os.path.join(args.output_dir, 'timing_results.csv')

with open(out_txt, 'w') as f:
    f.write(f"CDR Benchmark  –  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 90 + "\n")
    f.write(row_fmt.format(*headers) + "\n")
    f.write("  " + "─" * 82 + "\n")
    for r in records:
        rmse_str = f"{r['rmse']:.4f}" if not np.isnan(r['rmse']) else "N/A"
        f.write(row_fmt.format(
            r['dataset'][:42], r['model'], str(r['epochs']),
            rmse_str, r['time_fmt']) + "\n")

with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'model', 'epochs', 'rmse', 'time_s', 'time_fmt', 'status'])
    writer.writeheader()
    writer.writerows(records)

print(f"\n  Results saved to:\n    {out_txt}\n    {out_csv}")
print("=" * 90)
