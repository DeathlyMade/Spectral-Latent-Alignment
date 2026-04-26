import numpy as np
from Data_Preprocessing import Mydata
from SLA import compute_spectral_embeddings
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
dataset = Mydata(r"d:\Recsys\ratings_Office_Products.csv", r"d:\Recsys\ratings_Movies_and_TV.csv", train=None, preprocessed=True)

S_data = dataset.S_data
T_data = dataset.T_data
train_indices = dataset.train_indices
test_indices = dataset.test_indices

def evaluate_sla(k):
    print(f"--- Evaluatiing k={k} ---")
    A = R.copy()
    U_T_user, U_T_item = compute_spectral_embeddings(T_data, k=k)
    
    # Align
    M = U_S_user.T @ U_T_user
    U_M, S_M, Vt_M = np.linalg.svd(M)
    Q = U_M @ Vt_M
    U_S_user_aligned = U_S_user @ Q
    
    # Train Data
    u_idx, i_idx = np.where(S_data[train_indices] > 0)
    real_u_idx = train_indices[u_idx]
    y_train = S_data[real_u_idx, i_idx]
    X_train = np.hstack((U_S_user_aligned[real_u_idx], U_S_item[i_idx]))
    
    # Test Data
    u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
    real_u_idx_test = test_indices[u_idx_test]
    y_test = T_data[real_u_idx_test, i_idx_test]
    X_test = np.hstack((U_T_user[real_u_idx_test], U_T_item[i_idx_test]))
    
    # Model 1: Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    preds_ridge = np.clip(ridge.predict(X_test), 1, 5)
    rmse_ridge = math.sqrt(mean_squared_error(y_test, preds_ridge))
    print(f"Ridge RMSE: {rmse_ridge:.4f}")
    
    # Model 2: HGBR
    hgbr = HistGradientBoostingRegressor(max_iter=100)
    hgbr.fit(X_train, y_train)
    preds_hgbr = np.clip(hgbr.predict(X_test), 1, 5)
    rmse_hgbr = math.sqrt(mean_squared_error(y_test, preds_hgbr))
    print(f"HGBR RMSE: {rmse_hgbr:.4f}")

for k in [50, 100, 200]:
    evaluate_sla(k)
