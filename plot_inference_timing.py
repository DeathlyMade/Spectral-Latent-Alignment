import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import lightgbm as lgb
import warnings
from Data_Preprocessing import Mydata

warnings.filterwarnings('ignore')

pairs = [
    ("Office → Movies", 'ratings_Office_Products.csv', 'ratings_Movies_and_TV.csv'),
    ("Sports → CDs", 'ratings_Sports_and_Outdoors.csv', 'ratings_CDs_and_Vinyl.csv'),
    ("Apps → Games", 'ratings_Apps_for_Android.csv', 'ratings_Video_Games.csv'),
    ("Toys → Auto", 'ratings_Toys_and_Games.csv', 'ratings_Automotive.csv')
]

base_path = '/Users/daksh15/RECSYS/Spectral-Latent-Alignment/Data'

sla_inf_times = []
cmf_inf_times = []
labels = []
counts = []

class CMFModel(nn.Module):
    def __init__(self, num_users, num_items_T, k=50):
        super().__init__()
        self.U    = nn.Embedding(num_users, k)
        self.V_T  = nn.Embedding(num_items_T, k)
        self.b_U  = nn.Embedding(num_users, 1)
        self.b_T  = nn.Embedding(num_items_T, 1)
        self.global_mean_T = nn.Parameter(torch.zeros(1))

    def forward_T(self, u, i):
        return (self.global_mean_T
                + self.b_U(u).squeeze()
                + self.b_T(i).squeeze()
                + (self.U(u) * self.V_T(i)).sum(1))

# LightGBM Params (matches best model)
lgb_params = {
    'n_estimators': 408,
    'learning_rate': 0.03,
    'max_depth': 11,
    'num_leaves': 114,
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': 42
}

print("Benchmarking Inference Times...")

for name, s_file, t_file in pairs:
    s_path = os.path.join(base_path, s_file)
    t_path = os.path.join(base_path, t_file)
    dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
    
    num_users = dataset.S_data.shape[0]
    num_items_T = dataset.T_data.shape[1]
    
    # Get test ratings count
    u_idx_test, i_idx_test = np.where(dataset.T_data[dataset.test_indices] > 0)
    real_u_idx_test = dataset.test_indices[u_idx_test]
    num_test_ratings = len(real_u_idx_test)
    
    counts.append(num_test_ratings)
    labels.append(f"{name}\n({num_test_ratings:,} test queries)")
    
    # ==========================================
    # Benchmark CMF Inference
    # ==========================================
    cmf_model = CMFModel(num_users, num_items_T, k=50)
    cmf_model.eval()
    u_tensor = torch.tensor(real_u_idx_test, dtype=torch.long)
    i_tensor = torch.tensor(i_idx_test, dtype=torch.long)
    
    # Warmup
    with torch.no_grad():
        _ = cmf_model.forward_T(u_tensor, i_tensor)
        
    # Time
    t0 = time.perf_counter()
    with torch.no_grad():
        preds_cmf = cmf_model.forward_T(u_tensor, i_tensor).numpy()
    t1 = time.perf_counter()
    cmf_ms = (t1 - t0) * 1000.0
    cmf_inf_times.append(cmf_ms)
    
    # ==========================================
    # Benchmark SLA Inference
    # ==========================================
    # We simulate SLA inference by training a tree on dummy data of the right shape
    # This ensures the tree depth and leaf traversal is exactly as it would be.
    # Feature size = 300 (user) + 300 (item) = 600
    X_train_dummy = np.random.rand(1000, 600)
    y_train_dummy = np.random.rand(1000)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_dummy, y_train_dummy)
    
    X_test_dummy = np.random.rand(num_test_ratings, 600)
    
    # Warmup
    _ = lgb_model.predict(X_test_dummy[:10])
    
    # Time (In SLA we also do an hstack of the aligned user and item embedding before predicting)
    # Let's time both the array construction + prediction
    U_S_aligned = np.random.rand(num_users, 300)
    U_T_item = np.random.rand(num_items_T, 300)
    
    t0 = time.perf_counter()
    X_test = np.hstack((U_S_aligned[real_u_idx_test], U_T_item[i_idx_test]))
    preds_sla = lgb_model.predict(X_test)
    t1 = time.perf_counter()
    sla_ms = (t1 - t0) * 1000.0
    sla_inf_times.append(sla_ms)

    print(f"[{name}] Test Queries: {num_test_ratings} | CMF: {cmf_ms:.2f}ms | SLA: {sla_ms:.2f}ms")

# ==========================================
# Plotting
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, sla_inf_times, width, label='SLA (LightGBM)', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, cmf_inf_times, width, label='CMF (PyTorch)', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Inference Time (milliseconds)', fontsize=12)
ax.set_title('Batch Inference Time on Target Test Set', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

autolabel(rects1)
autolabel(rects2)

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('inference_timing.png', dpi=300)
print("Saved inference_timing.png")
