import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import warnings

from Data_Preprocessing import Mydata
from CMF import CMF, get_train_loaders, evaluate

warnings.filterwarnings('ignore')

pairs = [
    ("Office Products -> Movies and TV", 'ratings_Office_Products.csv', 'ratings_Movies_and_TV.csv'),
    ("Sports and Outdoors -> CDs and Vinyls", 'ratings_Sports_and_Outdoors.csv', 'ratings_CDs_and_Vinyl.csv'),
    ("Android Apps -> Video Games", 'ratings_Apps_for_Android.csv', 'ratings_Video_Games.csv'),
    ("Toys and Games -> Automotive", 'ratings_Toys_and_Games.csv', 'ratings_Automotive.csv')
]

base_path = '/Users/daksh15/RECSYS/Spectral-Latent-Alignment/Data'

fractions = np.linspace(0.1, 1.0, 10)
results = {name: [] for name, _, _ in pairs}
x_axis = {name: [] for name, _, _ in pairs}

def run_cmf_fraction(dataset, anchor_indices, k=50, alpha=0.5, lr=0.01, wd=1e-3, epochs=30):
    loader_S, loader_T, mean_S, mean_T = get_train_loaders(dataset.S_data, dataset.T_data, anchor_indices)
    
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
            
        rmse = evaluate(model, dataset.T_data, dataset.test_indices)
        if rmse < best_rmse:
            best_rmse = rmse
            
    return best_rmse

for name, s_file, t_file in pairs:
    print(f"\n======================================")
    print(f"Processing CMF: {name}")
    s_path = os.path.join(base_path, s_file)
    t_path = os.path.join(base_path, t_file)
    
    dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
    train_indices = dataset.train_indices
    
    for frac in fractions:
        num_anchors = int(len(train_indices) * frac)
        x_axis[name].append(num_anchors)
        
        np.random.seed(42) # Consistent sampling
        anchor_indices = np.random.choice(train_indices, size=num_anchors, replace=False)
        
        # Train CMF fully for this fraction
        rmse = run_cmf_fraction(dataset, anchor_indices, k=50, alpha=0.5, lr=0.01, wd=1e-3, epochs=30)
        print(f"Frac: {frac*100:3.0f}% | Anchors: {num_anchors:4d} | RMSE: {rmse:.4f}")
        results[name].append(rmse)

# Plotting
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D']
for i, (name, _, _) in enumerate(pairs):
    plt.plot(fractions * 100, results[name], marker=markers[i], linewidth=2, label=name)

plt.title('CMF Performance vs. Number of Anchors', fontsize=14, fontweight='bold')
plt.xlabel('Percentage of Training Users used as Anchors (%)', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('cmf_anchors_rmse.png', dpi=300)
print("\nPlot saved as cmf_anchors_rmse.png")
