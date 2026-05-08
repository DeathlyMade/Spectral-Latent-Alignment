import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

pairs = [
    ("Office → Movies", 15433, 46041),
    ("Sports → CDs", 24991, 63091),
    ("Apps → Games", 7668, 8769),
    ("Toys → Auto", 16070, 11905)
]

sla_times = []
cmf_times = []
labels = []

# Dummy LightGBM parameters (matches our SLA)
lgb_params = {
    'n_estimators': 408,
    'learning_rate': 0.03,
    'max_depth': 11,
    'num_leaves': 114,
    'verbosity': -1,
    'n_jobs': -1,
}

print("Benchmarking Cold-Start Inference (Time to serve 1 completely new user)...")

for name, num_items_S, num_items_T in pairs:
    labels.append(f"{name}\n({num_items_T//1000}k target items)")
    
    # ---------------------------------------------------------
    # Setup Dummy CMF Parameters
    # ---------------------------------------------------------
    V_S = torch.randn(num_items_S, 50)
    V_T = torch.randn(num_items_T, 50)
    
    # Simulate a new user who rated 15 source items
    num_source_ratings = 15
    user_i_idx = torch.randint(0, num_items_S, (num_source_ratings,))
    user_r_true = torch.randn(num_source_ratings)
    
    # CMF Cold-Start Simulation
    def run_cmf_coldstart():
        u_new = nn.Parameter(torch.randn(1, 50))
        opt = optim.Adam([u_new], lr=0.01)
        crit = nn.MSELoss()
        
        # 50 Epochs of Gradient Descent to learn the embedding
        for _ in range(50):
            opt.zero_grad()
            preds = (u_new.expand(num_source_ratings, 50) * V_S[user_i_idx]).sum(1)
            loss = crit(preds, user_r_true)
            loss.backward()
            opt.step()
            
        # Predict all target items
        with torch.no_grad():
            preds_target = (u_new.expand(num_items_T, 50) * V_T).sum(1)
        return preds_target

    # ---------------------------------------------------------
    # Setup Dummy SLA Parameters
    # ---------------------------------------------------------
    S_S_inv = np.random.rand(300)
    V_S_np = np.random.rand(num_items_S, 300)
    Q = np.random.rand(300, 300)
    U_T_item = np.random.rand(num_items_T, 300)
    r_new = np.random.rand(num_items_S) # binary interaction vector
    
    # Train dummy LGBM so it has the right tree depth
    dummy_X = np.random.rand(100, 600)
    dummy_y = np.random.rand(100)
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(dummy_X, dummy_y)
    
    # SLA Cold-Start Simulation
    def run_sla_coldstart():
        # Fold-in SVD embedding
        u_S = (r_new @ V_S_np) * S_S_inv
        # Procrustes Alignment
        u_T = u_S @ Q
        # Predict all target items
        u_T_repeated = np.tile(u_T, (num_items_T, 1))
        X_test = np.hstack((u_T_repeated, U_T_item))
        preds_target = model.predict(X_test)
        return preds_target
        
    # ---------------------------------------------------------
    # Timing
    # ---------------------------------------------------------
    # Warmup
    _ = run_cmf_coldstart()
    _ = run_sla_coldstart()
    
    # Benchmark CMF (Average of 5 runs)
    t0 = time.perf_counter()
    for _ in range(5):
        run_cmf_coldstart()
    cmf_ms = ((time.perf_counter() - t0) / 5) * 1000.0
    cmf_times.append(cmf_ms)
    
    # Benchmark SLA (Average of 5 runs)
    t0 = time.perf_counter()
    for _ in range(5):
        run_sla_coldstart()
    sla_ms = ((time.perf_counter() - t0) / 5) * 1000.0
    sla_times.append(sla_ms)
    
    print(f"[{name}] CMF: {cmf_ms:.1f} ms | SLA: {sla_ms:.1f} ms")


# ==========================================
# Plotting
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, sla_times, width, label='SLA (Linear Algebra + LightGBM)', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, cmf_times, width, label='CMF (SGD Backprop, 50 epochs)', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Inference Time per User (milliseconds)', fontsize=12)
ax.set_title('Cold-Start Inference Time (Serving a Completely New User)', fontsize=14, fontweight='bold')
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
plt.savefig('coldstart_timing.png', dpi=300)
print("Saved coldstart_timing.png")
