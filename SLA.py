import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds

class MLPPredictor(nn.Module):
    def __init__(self, input_dim):
        super(MLPPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, u_emb, i_emb):
        x = torch.cat([u_emb, i_emb], dim=1)
        return self.net(x).squeeze(-1)

def compute_spectral_embeddings(R, k=100):
    """
    Computes spectral embeddings from the user-item interaction matrix.
    R: user-item rating matrix (dense or sparse)
    """
    # Convert to binary adjacency for structure
    A = (R > 0).astype(float)
    
    # Degrees
    d_u = np.array(A.sum(axis=1)).flatten()
    d_i = np.array(A.sum(axis=0)).flatten()
    
    d_u[d_u == 0] = 1.0 # avoid div by zero
    d_i[d_i == 0] = 1.0
    
    D_u_inv = diags(1.0 / np.sqrt(d_u))
    D_i_inv = diags(1.0 / np.sqrt(d_i))
    
    # Normalized interaction matrix M = D_u^{-1/2} A D_i^{-1/2}
    if not isinstance(A, coo_matrix):
        A = coo_matrix(A)
    
    M = D_u_inv @ A @ D_i_inv
    
    # Top-k singular vectors
    u, s, vt = svds(M, k=k)
    
    # The columns of u are eigenvectors for users, rows of vt are for items
    # We sort them by singular values in descending order
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    vt = vt[idx, :]
    
    # embeddings
    U_user = u
    U_item = vt.T
    return U_user, U_item

def train_sla(S_data, T_data, train_indices, test_indices, k=100, epochs=50, lr=0.005):
    """
    SLA Implementation
    S_data: source rating matrix (dense numpy array)
    T_data: target rating matrix (dense numpy array)
    train_indices: list of user indices for training
    test_indices: list of user indices for testing
    k: spectral dimension
    """
    print("Computing spectral embeddings for Source...")
    U_S_user, U_S_item = compute_spectral_embeddings(S_data, k=k)
    
    print("Computing spectral embeddings for Target...")
    U_T_user, U_T_item = compute_spectral_embeddings(T_data, k=k)
    
    num_users = S_data.shape[0]
    
    # 1. Anchor Correspondences (users are exactly aligned 0 to num_users-1)
    X = U_S_user  # Anchor source embeddings
    Y = U_T_user  # Anchor target embeddings
    
    # 2. Embedding Alignment via Orthogonal Procrustes
    M = X.T @ Y
    U_M, S_M, Vt_M = np.linalg.svd(M)
    Q = U_M @ Vt_M
    
    # Aligned source user embeddings
    U_S_user_aligned = X @ Q
    
    # 3. Train Predictor on Aligned Source
    model = MLPPredictor(input_dim=k * 2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Prepare training data from source
    # We use all interactions in S_data from train_indices
    u_idx, i_idx = np.where(S_data[train_indices] > 0)
    # the actual user index is train_indices[u_idx]
    real_u_idx = train_indices[u_idx]
    ratings = S_data[real_u_idx, i_idx]
    
    train_dataset = TensorDataset(
        torch.tensor(U_S_user_aligned[real_u_idx], dtype=torch.float32),
        torch.tensor(U_S_item[i_idx], dtype=torch.float32),
        torch.tensor(ratings, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print("Training predictor on Source domain...")
    model.train()
    for ep in range(epochs):
        total_loss = 0
        for b_u, b_i, b_r in train_loader:
            optimizer.zero_grad()
            preds = model(b_u, b_i)
            loss = criterion(preds, b_r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_r)
        
        if ep % 10 == 0:
            print(f"Epoch {ep}, Source Loss: {total_loss / len(train_dataset):.4f}")
            
    # 4. Evaluate on Target Domain
    # The target uses U_T_user and U_T_item
    model.eval()
    u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
    real_u_idx_test = test_indices[u_idx_test]
    target_ratings = T_data[real_u_idx_test, i_idx_test]
    
    with torch.no_grad():
        test_u = torch.tensor(U_T_user[real_u_idx_test], dtype=torch.float32)
        test_i = torch.tensor(U_T_item[i_idx_test], dtype=torch.float32)
        preds = model(test_u, test_i).numpy()
        
        # clip predictions
        preds = np.clip(preds, 1.0, 5.0)
        rmse = np.sqrt(np.mean((preds - target_ratings)**2))
        
    return rmse
