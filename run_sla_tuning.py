import numpy as np
import math
import warnings
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

import sys
sys.path.append(r"d:\Recsys")
from Data_Preprocessing import Mydata

def compute_spectral_embeddings(R, k=100, use_ratings=True):
    if use_ratings:
        A = R.copy()
    else:
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
    
    # ensure k is smaller than min dimension
    actual_k = min(k, min(M.shape) - 1)
    u, s, vt = svds(M, k=actual_k)
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    vt = vt[idx, :]
    
    return u, vt.T

class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super(DeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, u_emb, i_emb):
        x = torch.cat([u_emb, i_emb], dim=1)
        return self.net(x).squeeze(-1)

print("Loading data for Grid Search (Office Products -> Movies and TV)...")
dataset = Mydata(r"d:\Recsys\ratings_Office_Products.csv", r"d:\Recsys\ratings_Movies_and_TV.csv", train=None, preprocessed=True)

S_data = dataset.S_data
T_data = dataset.T_data
train_indices = dataset.train_indices
test_indices = dataset.test_indices

results = []

k_values = [50, 100, 250, 500]

for k in k_values:
    print(f"\n====================== Evaluating k={k} ======================")
    for use_ratings in [False, True]:
        print(f"--- Feature Extractor: A = {'R' if use_ratings else '(R>0)'} ---")
        U_S_user, U_S_item = compute_spectral_embeddings(S_data, k=k, use_ratings=use_ratings)
        U_T_user, U_T_item = compute_spectral_embeddings(T_data, k=k, use_ratings=use_ratings)
        
        # Procrustes Alignment
        M = U_S_user.T @ U_T_user
        U_M, S_M, Vt_M = np.linalg.svd(M)
        Q = U_M @ Vt_M
        U_S_user_aligned = U_S_user @ Q
        
        u_idx, i_idx = np.where(S_data[train_indices] > 0)
        real_u_idx = train_indices[u_idx]
        y_train = S_data[real_u_idx, i_idx]
        X_train = np.hstack((U_S_user_aligned[real_u_idx], U_S_item[i_idx]))
        
        u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
        real_u_idx_test = test_indices[u_idx_test]
        y_test = T_data[real_u_idx_test, i_idx_test]
        X_test = np.hstack((U_T_user[real_u_idx_test], U_T_item[i_idx_test]))
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models = {
            "Ridge": Ridge(alpha=1.0),
            "HGBR": HistGradientBoostingRegressor(max_iter=100),
            "RandomForest": RandomForestRegressor(n_estimators=50, n_jobs=-1, max_depth=10)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_test), 1, 5)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            print(f"  {name} RMSE: {rmse:.4f}")
            results.append((k, use_ratings, name, rmse))
            
        # Deep MLP
        print(f"  Training Deep MLP...")
        mlp = DeepMLP(input_dim=U_S_user_aligned.shape[1] + U_S_item.shape[1])
        optimizer = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        train_dataset_mlp = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset_mlp, batch_size=256, shuffle=True)
        
        for ep in range(50):
            for b_x, b_y in train_loader:
                optimizer.zero_grad()
                preds = mlp(b_x[:, :U_S_user_aligned.shape[1]], b_x[:, U_S_user_aligned.shape[1]:])
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            preds_mlp = mlp(torch.tensor(X_test[:, :U_S_user_aligned.shape[1]], dtype=torch.float32), 
                            torch.tensor(X_test[:, U_S_user_aligned.shape[1]:], dtype=torch.float32)).numpy()
            preds_mlp = np.clip(preds_mlp, 1, 5)
            rmse_mlp = math.sqrt(mean_squared_error(y_test, preds_mlp))
        print(f"  DeepMLP RMSE: {rmse_mlp:.4f}")
        results.append((k, use_ratings, "DeepMLP", rmse_mlp))
        
        # Save results iteratively so we don't lose them if it hangs
        with open("tuning_progress.txt", "a") as f:
            f.write(f"k={k}, Ratings={use_ratings} -> Ridge: {results[-3][3]:.4f}, HGBR: {results[-2][3]:.4f}, RF: {results[-1][3]:.4f}, DeepMLP: {rmse_mlp:.4f}\n")

print("\n\nGrid Search Summary (Target to beat: I-DARec = 0.4697):")
results.sort(key=lambda x: x[3])
for r in results:
    print(f"k={r[0]}, Ratings={r[1]}, Model={r[2]} -> RMSE: {r[3]:.4f}")
