import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import math
import os
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds

from Data_Preprocessing import Mydata

warnings.filterwarnings('ignore')

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
    
    try:
        u, s, vt = svds(M, k=k, maxiter=100000)
    except Exception as e:
        print(f"SVD failed for k={k}. Trying k={k//2}...")
        try:
            u, s, vt = svds(M, k=k//2, maxiter=200000)
        except Exception as e2:
            print("Fallback SVD failed, using minimal k=50...")
            u, s, vt = svds(M, k=50, maxiter=500000)

    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    vt = vt[idx, :]
    return u, vt.T

pairs = [
    ("Office Products -> Movies and TV", 'ratings_Office_Products.csv', 'ratings_Movies_and_TV.csv'),
    ("Sports and Outdoors -> CDs and Vinyls", 'ratings_Sports_and_Outdoors.csv', 'ratings_CDs_and_Vinyl.csv'),
    ("Android Apps -> Video Games", 'ratings_Apps_for_Android.csv', 'ratings_Video_Games.csv'),
    ("Toys and Games -> Automotive", 'ratings_Toys_and_Games.csv', 'ratings_Automotive.csv')
]

base_path = '/Users/daksh15/RECSYS/Spectral-Latent-Alignment/Data'

k = 300
lgb_params = {
    'n_estimators': 408,
    'learning_rate': 0.031180345586184193,
    'max_depth': 11,
    'num_leaves': 114,
    'subsample': 0.7553865456047926,
    'colsample_bytree': 0.8985548992684733,
    'reg_alpha': 0.025839493370082733,
    'reg_lambda': 0.00011147011126221993,
    'min_child_samples': 36,
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': 42
}

fractions = np.linspace(0.1, 1.0, 10)
results = {name: [] for name, _, _ in pairs}
x_axis = {name: [] for name, _, _ in pairs}

for name, s_file, t_file in pairs:
    print(f"\n======================================")
    print(f"Processing: {name}")
    s_path = os.path.join(base_path, s_file)
    t_path = os.path.join(base_path, t_file)
    
    dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
    S_data = dataset.S_data
    T_data = dataset.T_data
    train_indices = dataset.train_indices
    test_indices = dataset.test_indices
    
    print(f"Computing SVDs with k={k}...")
    U_S_user, U_S_item = compute_spectral_embeddings(S_data, k=k)
    U_T_user, U_T_item = compute_spectral_embeddings(T_data, k=k)
    
    # Train the predictor ONCE per dataset using ALL train target data
    # (Since LightGBM only maps target_user to target_rating, it doesn't use anchors)
    u_idx, i_idx = np.where(T_data[train_indices] > 0)
    real_u_idx = train_indices[u_idx]
    y_train = T_data[real_u_idx, i_idx]
    X_train = np.hstack((U_T_user[real_u_idx], U_T_item[i_idx]))
    
    print("Training LightGBM predictor...")
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)
    
    # Get test ground truth
    u_idx_test, i_idx_test = np.where(T_data[test_indices] > 0)
    real_u_idx_test = test_indices[u_idx_test]
    y_test = T_data[real_u_idx_test, i_idx_test]
    
    # Vary the number of anchors
    for frac in fractions:
        num_anchors = int(len(train_indices) * frac)
        x_axis[name].append(num_anchors)
        
        # Sample anchors randomly from train_indices
        np.random.seed(42) # Fixed seed ensures smaller fractions are subsets
        anchor_indices = np.random.choice(train_indices, size=num_anchors, replace=False)
        
        U_S_anchors = U_S_user[anchor_indices]
        U_T_anchors = U_T_user[anchor_indices]
        
        # Align embeddings using ONLY the anchors
        M_mat = U_S_anchors.T @ U_T_anchors
        U_M, _, Vt_M = np.linalg.svd(M_mat)
        Q = U_M @ Vt_M
        
        # Align ALL source users using the Q derived from anchors
        U_S_user_aligned = U_S_user @ Q
        
        # Test on the test set using the aligned source user embeddings
        X_test = np.hstack((U_S_user_aligned[real_u_idx_test], U_T_item[i_idx_test]))
        preds = np.clip(model.predict(X_test), 1.0, 5.0)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        
        print(f"Frac: {frac*100:3.0f}% | Anchors: {num_anchors:4d} | RMSE: {rmse:.4f}")
        results[name].append(rmse)

# Plotting
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D']
for i, (name, _, _) in enumerate(pairs):
    plt.plot(fractions * 100, results[name], marker=markers[i], linewidth=2, label=name)

plt.title('SLA Performance vs. Number of Anchors', fontsize=14, fontweight='bold')
plt.xlabel('Percentage of Training Users used as Anchors (%)', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('anchors_rmse.png', dpi=300)
print("\nPlot saved as anchors_rmse.png")
