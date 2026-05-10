"""
Alignment Gap Evaluation
========================
Measures the true cross-domain learning quantification for both SLA and CMF.

Alignment Gap = RMSE(Cross-Domain) - RMSE(Oracle Theoretical Limit)

- SLA Oracle:  Predictor uses TRUE target user embeddings + target item embeddings
- SLA Cross:   Predictor uses ALIGNED source user embeddings + target item embeddings
- CMF Oracle:  Target-only MF trained on ALL target data (including test users)
- CMF Cross:   Joint CMF trained on Source + Target(train users only)
"""

import os
import sys
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

warnings.filterwarnings('ignore')

from Data_Preprocessing import Mydata

# ============================================================
# Spectral Embedding (SLA)
# ============================================================
def compute_spectral_embeddings(R, k=200):
    A = (R > 0).astype(float)
    d_u = np.array(A.sum(axis=1)).flatten()
    d_i = np.array(A.sum(axis=0)).flatten()
    d_u[d_u == 0] = 1.0
    d_i[d_i == 0] = 1.0
    D_u_inv = diags(1.0 / np.sqrt(d_u))
    D_i_inv = diags(1.0 / np.sqrt(d_i))
    if not isinstance(A, coo_matrix):
        A = coo_matrix(A)
    M = D_u_inv @ A @ D_i_inv
    actual_k = min(k, min(M.shape) - 2)
    try:
        u, s, vt = svds(M, k=actual_k, maxiter=100000)
    except Exception:
        actual_k = min(actual_k, min(M.shape) // 10)
        u, s, vt = svds(M, k=actual_k, maxiter=200000)
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    vt = vt[idx, :]
    return u, vt.T

# ============================================================
# CMF Model
# ============================================================
class CMF(nn.Module):
    def __init__(self, num_users, num_items_S, num_items_T, k=50):
        super(CMF, self).__init__()
        self.U = nn.Embedding(num_users, k)
        self.V_S = nn.Embedding(num_items_S, k)
        self.V_T = nn.Embedding(num_items_T, k)
        self.b_U = nn.Embedding(num_users, 1)
        self.b_S = nn.Embedding(num_items_S, 1)
        self.b_T = nn.Embedding(num_items_T, 1)
        self.global_mean_S = nn.Parameter(torch.zeros(1))
        self.global_mean_T = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V_S.weight, std=0.01)
        nn.init.normal_(self.V_T.weight, std=0.01)
        nn.init.zeros_(self.b_U.weight)
        nn.init.zeros_(self.b_S.weight)
        nn.init.zeros_(self.b_T.weight)

    def forward_S(self, users, items):
        dot = (self.U(users) * self.V_S(items)).sum(1)
        return self.global_mean_S + self.b_U(users).squeeze() + self.b_S(items).squeeze() + dot

    def forward_T(self, users, items):
        dot = (self.U(users) * self.V_T(items)).sum(1)
        return self.global_mean_T + self.b_U(users).squeeze() + self.b_T(items).squeeze() + dot

# Target-only MF (for CMF Oracle)
class TargetMF(nn.Module):
    def __init__(self, num_users, num_items, k=50):
        super(TargetMF, self).__init__()
        self.U = nn.Embedding(num_users, k)
        self.V = nn.Embedding(num_items, k)
        self.b_U = nn.Embedding(num_users, 1)
        self.b_V = nn.Embedding(num_items, 1)
        self.global_mean = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.zeros_(self.b_U.weight)
        nn.init.zeros_(self.b_V.weight)

    def forward(self, users, items):
        dot = (self.U(users) * self.V(items)).sum(1)
        return self.global_mean + self.b_U(users).squeeze() + self.b_V(items).squeeze() + dot

# ============================================================
# CMF Training Helpers
# ============================================================
def get_cmf_train_loaders(S_data, T_data, train_indices, batch_size=512):
    u_idx_S, i_idx_S = np.where(S_data > 0)
    ratings_S = S_data[u_idx_S, i_idx_S]

    T_train_data = np.zeros_like(T_data)
    T_train_data[train_indices] = T_data[train_indices]
    u_idx_T, i_idx_T = np.where(T_train_data > 0)
    ratings_T = T_data[u_idx_T, i_idx_T]

    dataset_S = TensorDataset(torch.tensor(u_idx_S, dtype=torch.long),
                              torch.tensor(i_idx_S, dtype=torch.long),
                              torch.tensor(ratings_S, dtype=torch.float32))
    dataset_T = TensorDataset(torch.tensor(u_idx_T, dtype=torch.long),
                              torch.tensor(i_idx_T, dtype=torch.long),
                              torch.tensor(ratings_T, dtype=torch.float32))
    loader_S = DataLoader(dataset_S, batch_size=batch_size, shuffle=True)
    loader_T = DataLoader(dataset_T, batch_size=batch_size, shuffle=True)
    return loader_S, loader_T, ratings_S.mean(), ratings_T.mean()

def evaluate_cmf(model, T_data, test_indices):
    model.eval()
    u_idx_T, i_idx_T = np.where(T_data[test_indices] > 0)
    real_u_idx = test_indices[u_idx_T]
    target_ratings = T_data[real_u_idx, i_idx_T]
    with torch.no_grad():
        preds = model.forward_T(torch.tensor(real_u_idx, dtype=torch.long),
                                torch.tensor(i_idx_T, dtype=torch.long)).numpy()
        preds = np.clip(preds, 1.0, 5.0)
        rmse = np.sqrt(np.mean((preds - target_ratings)**2))
    return rmse

def train_cmf_cross_domain(S_data, T_data, train_indices, test_indices,
                           k=50, alpha=0.5, lr=0.01, wd=1e-3, epochs=30):
    """Train standard CMF (cross-domain). Test users are cold-start in target."""
    loader_S, loader_T, mean_S, mean_T = get_cmf_train_loaders(S_data, T_data, train_indices)
    num_users = S_data.shape[0]
    num_items_S = S_data.shape[1]
    num_items_T = T_data.shape[1]

    model = CMF(num_users, num_items_S, num_items_T, k=k)
    with torch.no_grad():
        model.global_mean_S.fill_(mean_S)
        model.global_mean_T.fill_(mean_T)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_rmse = float('inf')

    for ep in range(epochs):
        model.train()
        iter_S = iter(loader_S)
        iter_T = iter(loader_T)
        num_batches = max(len(loader_S), len(loader_T))
        for _ in range(num_batches):
            optimizer.zero_grad()
            try:
                b_u_S, b_i_S, b_r_S = next(iter_S)
            except StopIteration:
                iter_S = iter(loader_S)
                b_u_S, b_i_S, b_r_S = next(iter_S)
            preds_S = model.forward_S(b_u_S, b_i_S)
            loss_S = criterion(preds_S, b_r_S)
            try:
                b_u_T, b_i_T, b_r_T = next(iter_T)
            except StopIteration:
                iter_T = iter(loader_T)
                b_u_T, b_i_T, b_r_T = next(iter_T)
            preds_T = model.forward_T(b_u_T, b_i_T)
            loss_T = criterion(preds_T, b_r_T)
            total_loss = alpha * loss_S + (1 - alpha) * loss_T
            total_loss.backward()
            optimizer.step()
        rmse = evaluate_cmf(model, T_data, test_indices)
        if rmse < best_rmse:
            best_rmse = rmse
    return best_rmse

def train_cmf_oracle(T_data, test_indices, k=50, lr=0.01, wd=1e-3, epochs=30):
    """
    CMF Oracle: Target-only MF trained on ALL target data (including test users).
    Evaluates reconstruction error on test users' ratings.
    """
    num_users = T_data.shape[0]
    num_items = T_data.shape[1]

    # Use ALL target interactions for training (oracle has full access)
    u_idx, i_idx = np.where(T_data > 0)
    ratings = T_data[u_idx, i_idx]
    mean_r = ratings.mean()

    dataset = TensorDataset(torch.tensor(u_idx, dtype=torch.long),
                            torch.tensor(i_idx, dtype=torch.long),
                            torch.tensor(ratings, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = TargetMF(num_users, num_items, k=k)
    with torch.no_grad():
        model.global_mean.fill_(mean_r)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_rmse = float('inf')

    for ep in range(epochs):
        model.train()
        for b_u, b_i, b_r in loader:
            optimizer.zero_grad()
            preds = model(b_u, b_i)
            loss = criterion(preds, b_r)
            loss.backward()
            optimizer.step()

        # Evaluate on test users' ratings
        model.eval()
        u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
        real_u_idx_test = test_indices[u_idx_test]
        target_ratings = T_data[real_u_idx_test, i_idx_test]
        with torch.no_grad():
            preds = model(torch.tensor(real_u_idx_test, dtype=torch.long),
                          torch.tensor(i_idx_test, dtype=torch.long)).numpy()
            preds = np.clip(preds, 1.0, 5.0)
            rmse = np.sqrt(np.mean((preds - target_ratings)**2))
        if rmse < best_rmse:
            best_rmse = rmse
    return best_rmse

# ============================================================
# SLA Evaluation
# ============================================================
def run_sla_evaluation(S_data, T_data, train_indices, test_indices, k=200):
    """Returns (oracle_rmse, cross_domain_rmse)"""
    print("  [SLA] Computing spectral embeddings...")
    U_S_user, U_S_item = compute_spectral_embeddings(S_data, k=k)
    U_T_user, U_T_item = compute_spectral_embeddings(T_data, k=k)

    # Procrustes alignment
    M_mat = U_S_user.T @ U_T_user
    U_M, _, Vt_M = np.linalg.svd(M_mat)
    Q = U_M @ Vt_M
    U_S_user_aligned = U_S_user @ Q

    # Training data: target user embs + target item embs -> target ratings (train users)
    u_idx, i_idx = np.where(T_data[train_indices] > 0)
    real_u_idx = train_indices[u_idx]
    y_train = T_data[real_u_idx, i_idx]
    X_train = np.hstack((U_T_user[real_u_idx], U_T_item[i_idx]))

    # Test data
    u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
    real_u_idx_test = test_indices[u_idx_test]
    y_test = T_data[real_u_idx_test, i_idx_test]

    # Oracle test: TRUE target user embeddings
    X_test_oracle = np.hstack((U_T_user[real_u_idx_test], U_T_item[i_idx_test]))
    # Cross-domain test: ALIGNED source user embeddings
    X_test_cross = np.hstack((U_S_user_aligned[real_u_idx_test], U_T_item[i_idx_test]))

    lgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.01,
        'max_depth': 7,
        'num_leaves': 100,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'reg_alpha': 1e-5,
        'reg_lambda': 0.05,
        'min_child_samples': 12,
        'verbosity': -1,
        'n_jobs': -1,
    }

    print("  [SLA] Training LightGBM predictor...")
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)

    # Oracle RMSE
    preds_oracle = np.clip(model.predict(X_test_oracle), 1.0, 5.0)
    oracle_rmse = math.sqrt(mean_squared_error(y_test, preds_oracle))

    # Cross-domain RMSE
    preds_cross = np.clip(model.predict(X_test_cross), 1.0, 5.0)
    cross_rmse = math.sqrt(mean_squared_error(y_test, preds_cross))

    return oracle_rmse, cross_rmse

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    pairs = [
        ("Office Products -> Movies and TV",
         r"./Data/ratings_Office_Products.csv", r"./Data/ratings_Movies_and_TV.csv"),
        ("Sports and Outdoors -> CDs and Vinyls",
         r"./Data/ratings_Sports_and_Outdoors.csv", r"./Data/ratings_CDs_and_Vinyl.csv"),
        ("Android Apps -> Video Games",
         r"./Data/ratings_Apps_for_Android.csv", r"./Data/ratings_Video_Games.csv"),
        ("Toys and Games -> Automotive",
         r"./Data/ratings_Toys_and_Games.csv", r"./Data/ratings_Automotive.csv"),
    ]

    all_results = []

    for name, s_path, t_path in pairs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Load data — regenerate .npy for this specific pair
        print("  Loading data (regenerating for this pair)...")
        dataset = Mydata(s_path, t_path, train=None, preprocessed=False)
        S_data = dataset.S_data if isinstance(dataset.S_data, np.ndarray) else dataset.S_data.numpy()
        T_data = dataset.T_data if isinstance(dataset.T_data, np.ndarray) else dataset.T_data.numpy()
        train_indices = dataset.train_indices
        test_indices = dataset.test_indices
        print(f"  Users: {S_data.shape[0]}, S_items: {S_data.shape[1]}, T_items: {T_data.shape[1]}")
        print(f"  Train users: {len(train_indices)}, Test users: {len(test_indices)}")

        # --- SLA ---
        print("\n  --- SLA ---")
        sla_oracle, sla_cross = run_sla_evaluation(S_data, T_data, train_indices, test_indices, k=200)
        sla_gap = sla_cross - sla_oracle
        print(f"  SLA Oracle RMSE:       {sla_oracle:.4f}")
        print(f"  SLA Cross-Domain RMSE: {sla_cross:.4f}")
        print(f"  SLA Alignment Gap:     {sla_gap:.4f}")

        # --- CMF ---
        print("\n  --- CMF ---")
        print("  [CMF] Training cross-domain model...")
        cmf_cross = train_cmf_cross_domain(S_data, T_data, train_indices, test_indices,
                                            k=50, alpha=0.5, lr=0.01, wd=1e-3, epochs=30)
        print(f"  CMF Cross-Domain RMSE: {cmf_cross:.4f}")

        print("  [CMF] Training oracle model (target-only MF)...")
        cmf_oracle = train_cmf_oracle(T_data, test_indices, k=50, lr=0.01, wd=1e-3, epochs=30)
        cmf_gap = cmf_cross - cmf_oracle
        print(f"  CMF Oracle RMSE:       {cmf_oracle:.4f}")
        print(f"  CMF Alignment Gap:     {cmf_gap:.4f}")

        all_results.append({
            'name': name,
            'sla_oracle': sla_oracle,
            'sla_cross': sla_cross,
            'sla_gap': sla_gap,
            'cmf_oracle': cmf_oracle,
            'cmf_cross': cmf_cross,
            'cmf_gap': cmf_gap,
        })

    # ============================================================
    # Summary
    # ============================================================
    print("\n\n" + "="*80)
    print("ALIGNMENT GAP RESULTS SUMMARY")
    print("="*80)
    header = f"{'Dataset Pair':<40} {'SLA Oracle':>10} {'SLA Cross':>10} {'SLA Gap':>10} {'CMF Oracle':>10} {'CMF Cross':>10} {'CMF Gap':>10}"
    print(header)
    print("-"*len(header))
    for r in all_results:
        print(f"{r['name']:<40} {r['sla_oracle']:>10.4f} {r['sla_cross']:>10.4f} {r['sla_gap']:>10.4f} {r['cmf_oracle']:>10.4f} {r['cmf_cross']:>10.4f} {r['cmf_gap']:>10.4f}")

    with open("alignment_gap_results.txt", "w") as f:
        f.write("ALIGNMENT GAP RESULTS\n")
        f.write("="*80 + "\n")
        f.write("Alignment Gap = RMSE(Cross-Domain) - RMSE(Oracle Theoretical Limit)\n")
        f.write("A LOWER gap means better cross-domain transfer.\n\n")
        f.write(header + "\n")
        f.write("-"*len(header) + "\n")
        for r in all_results:
            f.write(f"{r['name']:<40} {r['sla_oracle']:>10.4f} {r['sla_cross']:>10.4f} {r['sla_gap']:>10.4f} {r['cmf_oracle']:>10.4f} {r['cmf_cross']:>10.4f} {r['cmf_gap']:>10.4f}\n")

    print(f"\nResults saved to alignment_gap_results.txt")
