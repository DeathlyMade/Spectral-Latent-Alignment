import os
import sys
import math
import warnings
import numpy as np
import optuna
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

warnings.filterwarnings('ignore')

from Data_Preprocessing import Mydata

def compute_spectral_embeddings(R, k=100):
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
        actual_k = min(actual_k, min(M.shape) // 10)
        u, s, vt = svds(M, k=actual_k, maxiter=20000)
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    vt = vt[idx, :]
    return u, vt.T, s

print("Loading Toys and Games -> Video Games data...")
s_file = r"./Data/ratings_Toys_and_Games.csv"
t_file = r"./Data/ratings_Video_Games.csv"
dataset = Mydata(s_file, t_file, train=None, preprocessed=True)
S_data = dataset.S_data
T_data = dataset.T_data
train_indices = dataset.train_indices
test_indices = dataset.test_indices

K_VALUES = [50, 100, 200, 300]
PRECOMPUTED = {}
for k in K_VALUES:
    print(f"Precomputing for k={k}...")
    U_S_user, U_S_item, s_S = compute_spectral_embeddings(S_data, k=k)
    U_T_user, U_T_item, s_T = compute_spectral_embeddings(T_data, k=k)
    
    M_mat = U_S_user.T @ U_T_user
    U_M, _, Vt_M = np.linalg.svd(M_mat)
    Q = U_M @ Vt_M
    U_S_user_aligned = U_S_user @ Q
    
    u_idx, i_idx = np.where(T_data[train_indices] > 0)
    real_u_idx = train_indices[u_idx]
    y_train = T_data[real_u_idx, i_idx]
    X_train = np.hstack((U_T_user[real_u_idx], U_T_item[i_idx]))
    
    u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
    real_u_idx_test = test_indices[u_idx_test]
    y_test = T_data[real_u_idx_test, i_idx_test]
    X_test = np.hstack((U_S_user_aligned[real_u_idx_test], U_T_item[i_idx_test]))
    
    PRECOMPUTED[k] = {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test
    }

def objective(trial):
    k = trial.suggest_categorical('k', K_VALUES)
    data = PRECOMPUTED[k]
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    n_estimators = trial.suggest_int('lgb_n_estimators', 50, 500)
    learning_rate = trial.suggest_float('lgb_lr', 1e-3, 1e-1, log=True)
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
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    return math.sqrt(mean_squared_error(y_test, preds))

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="lgb_toys_video")
    study.optimize(objective, n_trials=30) 
    
    print("\n" + "=" * 60)
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)
