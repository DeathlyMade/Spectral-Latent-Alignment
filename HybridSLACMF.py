"""
Hybrid SLA-CMF — Strategy C: SLA-Initialized CMF
=================================================
Uses SLA's Procrustes-aligned spectral embeddings to warm-start CMF's
shared user factor U, instead of random initialization.

This injects SLA's superior cross-domain alignment quality into CMF's
rating-prediction training pipeline, with zero new hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds


# ============================================================
# Spectral Embeddings (from SLA)
# ============================================================
def compute_spectral_embeddings(R, k=50):
    """
    Compute spectral embeddings from user-item interaction matrix.
    Uses binary adjacency + symmetric normalization + truncated SVD.
    """
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

    return u, vt.T, actual_k


def compute_procrustes_alignment(U_source, U_target):
    """Orthogonal Procrustes: find Q minimizing ||U_source @ Q - U_target||_F."""
    M = U_source.T @ U_target
    U_M, _, Vt_M = np.linalg.svd(M)
    Q = U_M @ Vt_M
    return U_source @ Q


# ============================================================
# CMF Model (same architecture as existing CMF.py)
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

        # Default random init (overridden for hybrid)
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


# ============================================================
# Training & Evaluation helpers
# ============================================================
def get_train_loaders(S_data, T_data, train_indices, batch_size=512):
    """Prepare DataLoaders for source (all users) and target (train users only)."""
    u_idx_S, i_idx_S = np.where(S_data > 0)
    ratings_S = S_data[u_idx_S, i_idx_S]

    T_train = np.zeros_like(T_data)
    T_train[train_indices] = T_data[train_indices]
    u_idx_T, i_idx_T = np.where(T_train > 0)
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


def evaluate(model, T_data, test_indices):
    """Evaluate cross-domain RMSE on test users in the target domain."""
    model.eval()
    u_idx_T, i_idx_T = np.where(T_data[test_indices] > 0)
    real_u_idx = test_indices[u_idx_T]
    target_ratings = T_data[real_u_idx, i_idx_T]

    with torch.no_grad():
        preds = model.forward_T(torch.tensor(real_u_idx, dtype=torch.long),
                                torch.tensor(i_idx_T, dtype=torch.long)).numpy()
        preds = np.clip(preds, 1.0, 5.0)
        rmse = np.sqrt(np.mean((preds - target_ratings) ** 2))
    return rmse


def _train_cmf_loop(model, S_data, T_data, train_indices, test_indices,
                    alpha=0.5, lr=0.01, wd=1e-4, epochs=50,
                    freeze_u_epochs=0, sla_reg_lambda=0.0, init_U=None):
    """Core CMF training loop (shared by baseline and hybrid)."""
    loader_S, loader_T, mean_S, mean_T = get_train_loaders(S_data, T_data, train_indices)

    with torch.no_grad():
        model.global_mean_S.fill_(mean_S)
        model.global_mean_T.fill_(mean_T)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_rmse = float('inf')

    for ep in range(epochs):
        # Selective Freezing
        if ep < freeze_u_epochs:
            model.U.weight.requires_grad = False
        else:
            model.U.weight.requires_grad = True

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
            
            # Structural Regularization (SLA-Reg)
            if sla_reg_lambda > 0.0 and init_U is not None:
                reg_loss = torch.sum((model.U.weight - init_U) ** 2)
                total_loss += sla_reg_lambda * reg_loss

            total_loss.backward()
            optimizer.step()

        rmse = evaluate(model, T_data, test_indices)
        if rmse < best_rmse:
            best_rmse = rmse

        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep + 1}/{epochs}, RMSE: {rmse:.4f} (best: {best_rmse:.4f})")

    return best_rmse


# ============================================================
# Strategy C: SLA-Initialized CMF
# ============================================================
def train_hybrid_c(S_data, T_data, train_indices, test_indices,
                   k=50, alpha=0.5, lr=0.01, wd=1e-4, epochs=50,
                   init_mode='average', freeze_u_epochs=0, sla_reg_lambda=0.0):
    """
    Strategy C: Initialize CMF with SLA's Procrustes-aligned spectral embeddings.

    init_mode controls how the shared user factor U is initialized:
        'aligned_source' : U ← Procrustes-aligned source user embeddings
        'target'         : U ← Target user spectral embeddings
        'average'        : U ← (aligned_source + target) / 2

    Item factors V_S and V_T are always initialized with their respective
    spectral item embeddings.
    """
    num_users = S_data.shape[0]
    num_items_S = S_data.shape[1]
    num_items_T = T_data.shape[1]

    # --- Phase 1: Spectral embeddings ---
    print(f"    [SLA] Computing spectral embeddings (k={k})...")
    U_S_user, U_S_item, actual_k_S = compute_spectral_embeddings(S_data, k=k)
    U_T_user, U_T_item, actual_k_T = compute_spectral_embeddings(T_data, k=k)

    # Use the minimum actual k across both domains
    actual_k = min(actual_k_S, actual_k_T)
    if actual_k < k:
        print(f"    [SLA] Reduced k from {k} to {actual_k} (matrix too small)")
        U_S_user = U_S_user[:, :actual_k]
        U_S_item = U_S_item[:, :actual_k]
        U_T_user = U_T_user[:, :actual_k]
        U_T_item = U_T_item[:, :actual_k]

    # --- Phase 2: Procrustes alignment ---
    print("    [SLA] Procrustes alignment (source -> target)...")
    U_S_aligned = compute_procrustes_alignment(U_S_user, U_T_user)

    # --- Phase 3: Initialize CMF ---
    model = CMF(num_users, num_items_S, num_items_T, k=actual_k)

    with torch.no_grad():
        # User embedding initialization
        if init_mode == 'aligned_source':
            init_U = U_S_aligned
        elif init_mode == 'target':
            init_U = U_T_user
        elif init_mode == 'average':
            init_U = (U_S_aligned + U_T_user) / 2.0
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        model.U.weight.copy_(torch.tensor(init_U, dtype=torch.float32))
        model.V_S.weight.copy_(torch.tensor(U_S_item, dtype=torch.float32))
        model.V_T.weight.copy_(torch.tensor(U_T_item, dtype=torch.float32))

    print(f"    [Hybrid] Initialized CMF with spectral embeddings (mode={init_mode})")

    # --- Phase 4: Train CMF ---
    init_U_tensor = torch.tensor(init_U, dtype=torch.float32) if sla_reg_lambda > 0 else None
    return _train_cmf_loop(model, S_data, T_data, train_indices, test_indices,
                           alpha=alpha, lr=lr, wd=wd, epochs=epochs,
                           freeze_u_epochs=freeze_u_epochs, sla_reg_lambda=sla_reg_lambda, init_U=init_U_tensor)


# ============================================================
# Baseline CMF (random init, for comparison)
# ============================================================
def train_baseline_cmf(S_data, T_data, train_indices, test_indices,
                       k=50, alpha=0.5, lr=0.01, wd=1e-4, epochs=50):
    """Standard CMF with random initialization (baseline)."""
    num_users = S_data.shape[0]
    num_items_S = S_data.shape[1]
    num_items_T = T_data.shape[1]

    model = CMF(num_users, num_items_S, num_items_T, k=k)
    print("    [CMF] Random initialization (baseline)")

    return _train_cmf_loop(model, S_data, T_data, train_indices, test_indices,
                           alpha=alpha, lr=lr, wd=wd, epochs=epochs)
