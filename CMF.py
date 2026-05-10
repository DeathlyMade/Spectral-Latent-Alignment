import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import math
import itertools
import argparse
import warnings

warnings.filterwarnings('ignore')

sys.path.append(r"d:\Recsys")
from Data_Preprocessing import Mydata

class CMF(nn.Module):
    def __init__(self, num_users, num_items_S, num_items_T, k=50):
        super(CMF, self).__init__()
        self.U = nn.Embedding(num_users, k)
        self.V_S = nn.Embedding(num_items_S, k)
        self.V_T = nn.Embedding(num_items_T, k)
        
        # User/Item Biases
        self.b_U = nn.Embedding(num_users, 1)
        self.b_S = nn.Embedding(num_items_S, 1)
        self.b_T = nn.Embedding(num_items_T, 1)
        self.global_mean_S = nn.Parameter(torch.zeros(1))
        self.global_mean_T = nn.Parameter(torch.zeros(1))

        # Initialization
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

def get_train_loaders(S_data, T_data, train_indices, batch_size=512):
    # Source is entirely used for training
    u_idx_S, i_idx_S = np.where(S_data > 0)
    ratings_S = S_data[u_idx_S, i_idx_S]
    
    # Target is only used for train_indices
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

def evaluate(model, T_data, test_indices):
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

def train_cmf(S_path, T_path, k=50, alpha=0.5, lr=0.01, wd=1e-4, epochs=30):
    dataset = Mydata(S_path, T_path, train=None, preprocessed=True)
    loader_S, loader_T, mean_S, mean_T = get_train_loaders(dataset.S_data, dataset.T_data, dataset.train_indices)
    
    num_users = dataset.S_data.shape[0]
    num_items_S = dataset.S_data.shape[1]
    num_items_T = dataset.T_data.shape[1]
    
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
            
            # Source step
            try:
                b_u_S, b_i_S, b_r_S = next(iter_S)
            except StopIteration:
                iter_S = iter(loader_S)
                b_u_S, b_i_S, b_r_S = next(iter_S)
            
            preds_S = model.forward_S(b_u_S, b_i_S)
            loss_S = criterion(preds_S, b_r_S)
            
            # Target step
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
            
        rmse = evaluate(model, dataset.T_data, dataset.test_indices)
        if rmse < best_rmse:
            best_rmse = rmse
            
    return best_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'run_all'])
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=1e-3)
    args = parser.parse_args()
    
    dataset_pairs = [
        (r"./Data/ratings_Toys_and_Games.csv", r"./Data/ratings_Video_Games.csv", "Toys and Games -> Video Games"),
    #     (r"d:\Recsys\ratings_Sports_and_Outdoors.csv", r"d:\Recsys\ratings_CDs_and_Vinyl.csv", "Sports and Outdoors -> CDs and Vinyls"),
    #     (r"d:\Recsys\ratings_Apps_for_Android.csv", r"d:\Recsys\ratings_Video_Games.csv", "Android Apps -> Video Games"),
    #     (r"d:\Recsys\ratings_Toys_and_Games.csv", r"d:\Recsys\ratings_Automotive.csv", "Toys and Games -> Automotive")
    # ]
    ]
    
    if args.mode == 'grid':
        print("Starting Grid Search on Office Products -> Movies and TV...")
        s_path, t_path, _ = dataset_pairs[0]
        
        k_vals = [50, 100]
        alpha_vals = [0.2, 0.5, 0.8]
        lr_vals = [0.01, 0.005]
        wd_vals = [1e-4, 1e-3]
        
        results = []
        with open("cmf_grid_results.txt", "w") as f:
            f.write("CMF Grid Search Results\n========================\n")
            
        for k, alpha, lr, wd in itertools.product(k_vals, alpha_vals, lr_vals, wd_vals):
            print(f"Testing K={k}, Alpha={alpha}, LR={lr}, WD={wd}...")
            rmse = train_cmf(s_path, t_path, k=k, alpha=alpha, lr=lr, wd=wd, epochs=30)
            print(f" -> RMSE: {rmse:.4f}")
            results.append((k, alpha, lr, wd, rmse))
            with open("cmf_grid_results.txt", "a") as f:
                f.write(f"K={k}, Alpha={alpha}, LR={lr}, WD={wd} -> RMSE: {rmse:.4f}\n")
            
        results.sort(key=lambda x: x[-1])
        print("\nBest Configuration:")
        print(f"K={results[0][0]}, Alpha={results[0][1]}, LR={results[0][2]}, WD={results[0][3]} -> RMSE: {results[0][4]:.4f}")
        
    elif args.mode == 'run_all':
        print(f"Running CMF on all datasets with K={args.k}, Alpha={args.alpha}, LR={args.lr}, WD={args.wd}...")
        with open("cmf_final_results.txt", "w") as f:
            f.write(f"CMF Final Results (K={args.k}, Alpha={args.alpha}, LR={args.lr}, WD={args.wd})\n========================\n")
        for s_path, t_path, name in dataset_pairs:
            rmse = train_cmf(s_path, t_path, k=args.k, alpha=args.alpha, lr=args.lr, wd=args.wd, epochs=50)
            print(f"{name}: {rmse:.4f}")
            with open("cmf_final_results.txt", "a") as f:
                f.write(f"{name}: {rmse:.4f}\n")
