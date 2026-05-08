import os
import sys
import math
import warnings
import numpy as np
import optuna
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# Import data processor
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from Data_Preprocessing import Mydata

# ---------------------------------------------------------------------------
# Utility: Spectral Embedding
# ---------------------------------------------------------------------------
def compute_spectral_embeddings(R, k=100):
    """Binary adjacency only (use_ratings=False has proven far superior)."""
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
        u, s, vt = svds(M, k=actual_k, maxiter=10000)
    except Exception:
        # Fallback: reduce k if convergence fails
        actual_k = min(actual_k, min(M.shape) // 10)
        u, s, vt = svds(M, k=actual_k, maxiter=20000)
    
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    vt = vt[idx, :]
    
    return u, vt.T, s

# ---------------------------------------------------------------------------
# MLP Definition
# ---------------------------------------------------------------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_units, dropout_rate):
        super(DeepMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Data Loading & Global Precomputation
# ---------------------------------------------------------------------------
def load_dataset():
    s_file = os.path.join(repo_root, "Data", "ratings_Office_Products.csv")
    t_file = os.path.join(repo_root, "Data", "ratings_Movies_and_TV.csv")
    
    try:
        dataset = Mydata(s_file, t_file, train=None, preprocessed=True)
    except Exception:
        print("Preprocessed .npy not found. Generating from CSV...")
        dataset = Mydata(s_file, t_file, train=None, preprocessed=False)
    
    S_data = dataset.S_data.numpy() if hasattr(dataset.S_data, 'numpy') else dataset.S_data
    T_data = dataset.T_data.numpy() if hasattr(dataset.T_data, 'numpy') else dataset.T_data
    train_indices = dataset.train_indices.numpy() if hasattr(dataset.train_indices, 'numpy') else dataset.train_indices
    test_indices = dataset.test_indices.numpy() if hasattr(dataset.test_indices, 'numpy') else dataset.test_indices
    
    return S_data, T_data, train_indices, test_indices

print("Loading data...")
S_data, T_data, train_indices, test_indices = load_dataset()

# Grid of k to precompute
K_VALUES = [50, 100, 150, 200, 250, 300]
print("Precomputing SVD & Alignments for K in", K_VALUES)

PRECOMPUTED = {}
for k in K_VALUES:
    print(f"  -> Precomputing k={k}")
    U_S_user, U_S_item, s_S = compute_spectral_embeddings(S_data, k=k)
    U_T_user, U_T_item, s_T = compute_spectral_embeddings(T_data, k=k)
    
    # Procrustes Alignment: align source users into target frame
    M_mat = U_S_user.T @ U_T_user
    U_M, _, Vt_M = np.linalg.svd(M_mat)
    Q = U_M @ Vt_M
    U_S_user_aligned = U_S_user @ Q  # Source users in target frame
    
    # ================================================================
    # Train: f(target_user, target_item) → target_rating  (clean signal)
    # Test:  f(aligned_source_user, target_item) → target_rating  (cross-domain transfer)
    # ================================================================
    
    # Train data: target user + target item → target rating (from TRAIN users)
    u_idx, i_idx = np.where(T_data[train_indices] > 0)
    real_u_idx = train_indices[u_idx]
    y_train = T_data[real_u_idx, i_idx]
    X_train = np.hstack((U_T_user[real_u_idx], U_T_item[i_idx]))
    
    # Test data: ALIGNED SOURCE user + target item → target rating (from TEST users)
    u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
    real_u_idx_test = test_indices[u_idx_test]
    y_test = T_data[real_u_idx_test, i_idx_test]
    X_test = np.hstack((U_S_user_aligned[real_u_idx_test], U_T_item[i_idx_test]))
    
    PRECOMPUTED[k] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'input_dim': X_train.shape[1],
    }

print(f"Precomputation complete. Train samples: {PRECOMPUTED[100]['X_train'].shape[0]}, "
      f"Test samples: {PRECOMPUTED[100]['X_test'].shape[0]}")

# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------
def objective(trial):
    k = trial.suggest_categorical('k', K_VALUES)
    
    data = PRECOMPUTED[k]
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    input_dim = data['input_dim']
    
    model_type = trial.suggest_categorical('model_type', ['DeepMLP', 'Ridge', 'HGBR', 'XGBoost', 'LightGBM'])
    
    if model_type == 'Ridge':
        alpha = trial.suggest_float('ridge_alpha', 1e-3, 1e3, log=True)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 1, 5)
        return math.sqrt(mean_squared_error(y_test, preds))
        
    elif model_type == 'HGBR':
        max_iter = trial.suggest_int('hgbr_max_iter', 50, 500)
        learning_rate = trial.suggest_float('hgbr_lr', 1e-3, 3e-1, log=True)
        max_depth = trial.suggest_int('hgbr_max_depth', 3, 15)
        min_samples_leaf = trial.suggest_int('hgbr_min_samples_leaf', 5, 50)
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 1, 5)
        return math.sqrt(mean_squared_error(y_test, preds))
        
    elif model_type == 'XGBoost':
        n_estimators = trial.suggest_int('xgb_n_estimators', 50, 500)
        learning_rate = trial.suggest_float('xgb_lr', 1e-3, 3e-1, log=True)
        max_depth = trial.suggest_int('xgb_max_depth', 3, 15)
        subsample = trial.suggest_float('xgb_subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('xgb_colsample', 0.5, 1.0)
        reg_alpha = trial.suggest_float('xgb_reg_alpha', 1e-6, 10.0, log=True)
        reg_lambda = trial.suggest_float('xgb_reg_lambda', 1e-6, 10.0, log=True)
        min_child_weight = trial.suggest_int('xgb_min_child_weight', 1, 20)
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, min_child_weight=min_child_weight,
            tree_method='hist', verbosity=0, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 1, 5)
        return math.sqrt(mean_squared_error(y_test, preds))
        
    elif model_type == 'LightGBM':
        n_estimators = trial.suggest_int('lgb_n_estimators', 50, 500)
        learning_rate = trial.suggest_float('lgb_lr', 1e-3, 3e-1, log=True)
        max_depth = trial.suggest_int('lgb_max_depth', 3, 15)
        num_leaves = trial.suggest_int('lgb_num_leaves', 15, 127)
        subsample = trial.suggest_float('lgb_subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('lgb_colsample', 0.5, 1.0)
        reg_alpha = trial.suggest_float('lgb_reg_alpha', 1e-6, 10.0, log=True)
        reg_lambda = trial.suggest_float('lgb_reg_lambda', 1e-6, 10.0, log=True)
        min_child_samples = trial.suggest_int('lgb_min_child_samples', 5, 50)
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, num_leaves=num_leaves,
            subsample=subsample, colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            min_child_samples=min_child_samples, verbosity=-1, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 1, 5)
        return math.sqrt(mean_squared_error(y_test, preds))
        
    elif model_type == 'DeepMLP':
        n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
        hidden_units = trial.suggest_categorical('mlp_hidden_units', [32, 64, 128, 256, 512])
        dropout_rate = trial.suggest_float('mlp_dropout', 0.0, 0.5)
        lr = trial.suggest_float('mlp_lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('mlp_weight_decay', 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('mlp_optimizer', ['Adam', 'AdamW'])
        batch_size = trial.suggest_categorical('mlp_batch_size', [128, 256, 512, 1024])
        
        mlp = DeepMLP(input_dim=input_dim, n_layers=n_layers,
                      hidden_units=hidden_units, dropout_rate=dropout_rate)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
            
        criterion = nn.MSELoss()
        
        # Split train into train/val for early stopping
        val_size = int(0.1 * len(X_train))
        perm = np.random.permutation(len(X_train))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        
        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        
        epochs = 150
        best_val_loss = float('inf')
        patience = 10
        stagnant = 0
        best_state = None
        
        for ep in range(epochs):
            mlp.train()
            for b_x, b_y in train_loader:
                optimizer.zero_grad()
                preds = mlp(b_x)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()
                
            mlp.eval()
            with torch.no_grad():
                val_preds = mlp(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
                stagnant = 0
            else:
                stagnant += 1
                
            if stagnant >= patience:
                break
        
        # Restore best model
        if best_state is not None:
            mlp.load_state_dict(best_state)
                
        # Final eval on test set
        mlp.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            preds_mlp = mlp(X_test_t).numpy()
            preds_mlp = np.clip(preds_mlp, 1, 5)
            rmse_mlp = math.sqrt(mean_squared_error(y_test, preds_mlp))
            
        return rmse_mlp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of optuna trials")
    args = parser.parse_args()
    
    study = optuna.create_study(direction="minimize", study_name="sla_hyperopt")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("Optimization Finished!")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"CMF Baseline: 0.9925")
    print(f"{'BEATS CMF!' if study.best_value < 0.9925 else 'Does not beat CMF yet.'}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    # Show top 5 trials
    print("\nTop 5 Trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, t in enumerate(trials_sorted[:5]):
        print(f"  #{i+1}: RMSE={t.value:.4f} | {t.params}")
    
    # Save best params to file
    with open("sla_best_optuna_params.txt", "w") as f:
        f.write(f"Best RMSE: {study.best_value:.4f}\n")
        f.write(f"CMF Baseline: 0.9925\n")
        f.write(f"{'BEATS CMF!' if study.best_value < 0.9925 else 'Does not beat CMF yet.'}\n\n")
        f.write("Best Params:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTop 5 Trials:\n")
        for i, t in enumerate(trials_sorted[:5]):
            f.write(f"  #{i+1}: RMSE={t.value:.4f} | {t.params}\n")
