"""
Runner: Hybrid SLA-CMF — Strategy C evaluation
================================================
Runs baseline CMF and three SLA-initialization variants across all 4 dataset pairs.
Reports RMSE only (no alignment gap).
"""

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Data_Preprocessing import Mydata
from HybridSLACMF import train_hybrid_c, train_baseline_cmf


if __name__ == "__main__":
    pairs = [
        ("Office Products -> Movies and TV",
         r"./Data/ratings_Office_Products.csv", r"./Data/ratings_Movies_and_TV.csv"),
        ("Sports and Outdoors -> CDs and Vinyls",
         r"./Data/ratings_Sports_and_Outdoors.csv", r"./Data/ratings_CDs_and_Vinyl.csv"),
        ("Android Apps -> Video Games",
         r"./Data/ratings_Apps_for_Android.csv", r"./Data/ratings_Video_Games.csv"),
        ("Toys and Games -> Automotive",
         r"./Data/ratings_Toys_and_Games.csv", r"./Data/ratings_Automotive.csv"),
    ]

    init_modes = ['aligned_source']
    all_results = []

    # CMF hyperparams (same as existing best)
    K = 50
    ALPHA = 0.5
    LR = 0.01
    WD = 1e-4
    EPOCHS = 50

    for name, s_path, t_path in pairs:
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")

        # Regenerate data for this specific pair (safe — avoids stale .npy)
        print("  Loading data (regenerating for this pair)...")
        dataset = Mydata(s_path, t_path, train=None, preprocessed=False)
        S_data = dataset.S_data if isinstance(dataset.S_data, np.ndarray) else dataset.S_data.numpy()
        T_data = dataset.T_data if isinstance(dataset.T_data, np.ndarray) else dataset.T_data.numpy()
        train_indices = dataset.train_indices
        test_indices = dataset.test_indices

        print(f"  Users: {S_data.shape[0]}, S_items: {S_data.shape[1]}, T_items: {T_data.shape[1]}")
        print(f"  Train users: {len(train_indices)}, Test users: {len(test_indices)}")

        result = {'name': name}

        # --- Baseline CMF (random init) ---
        print(f"\n  --- Baseline CMF (random init, k={K}) ---")
        cmf_rmse = train_baseline_cmf(S_data, T_data, train_indices, test_indices,
                                       k=K, alpha=ALPHA, lr=LR, wd=WD, epochs=EPOCHS)
        print(f"  >> CMF RMSE: {cmf_rmse:.4f}")
        result['cmf'] = cmf_rmse

        # --- Hybrid-C variants ---
        for mode in init_modes:
            print(f"\n  --- Hybrid-C (init={mode}, k={K}) ---")
            hybrid_rmse = train_hybrid_c(S_data, T_data, train_indices, test_indices,
                                          k=K, alpha=ALPHA, lr=LR, wd=WD, epochs=EPOCHS,
                                          init_mode=mode)
            print(f"  >> Hybrid-C ({mode}) RMSE: {hybrid_rmse:.4f}")
            result[f'hybrid_{mode}'] = hybrid_rmse

        all_results.append(result)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n\n{'=' * 90}")
    print("HYBRID SLA-CMF RESULTS (Strategy C: SLA-Initialized CMF)")
    print(f"{'=' * 90}")
 
    header = f"{'Dataset Pair':<45} {'CMF':>10} {'H-Aligned':>12} {'Improved?':>12}"
    print(header)
    print("-" * len(header))
 
    for r in all_results:
        cmf_val = r['cmf']
        hybrid_val = r['hybrid_aligned_source']
        improved = "YES" if hybrid_val < cmf_val else "no"
 
        print(f"{r['name']:<45} {cmf_val:>10.4f} {hybrid_val:>12.4f} {improved:>12}")
 
    # Save results
    with open("hybrid_results.txt", "w") as f:
        f.write("HYBRID SLA-CMF RESULTS (Strategy C: SLA-Initialized CMF)\n")
        f.write(f"Hyperparams: k={K}, alpha={ALPHA}, lr={LR}, wd={WD}, epochs={EPOCHS}\n")
        f.write("=" * 90 + "\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in all_results:
            cmf_val = r['cmf']
            hybrid_val = r['hybrid_aligned_source']
            improved = "YES" if hybrid_val < cmf_val else "no"
 
            f.write(f"{r['name']:<45} {cmf_val:>10.4f} {hybrid_val:>12.4f} {improved:>12}\n")
 
    print(f"\nResults saved to hybrid_results.txt")
