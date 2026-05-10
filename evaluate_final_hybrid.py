"""
Final Evaluator: Hybrid SLA-CMF (Strategy C) Across All 4 Dataset Pairs
=======================================================================
Uses the optimal hyperparameters discovered via Optuna tuning.
Compares the results directly against the historical CMF baseline results.
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
    # Hyperparameters from hybrid_best_params.txt
    K = 25
    ALPHA = 0.5933251873865484
    LR = 0.00031146806126948034
    WD = 1.5996074234205064e-06
    EPOCHS = 50

    # Historical Baseline CMF Results from 'cmf_final_results.txt'
    historical_cmf = {
        "Office Products -> Movies and TV": 0.9939,
        "Sports and Outdoors -> CDs and Vinyls": 0.8002,
        "Android Apps -> Video Games": 1.1367,
        "Toys and Games -> Automotive": 0.9452
    }

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

    all_results = []

    print("\n" + "=" * 85)
    print("STARTING FINAL HYBRID SLA-CMF EVALUATION ACROSS ALL 4 DATASET PAIRS")
    print("=" * 85)
    print(f"Using Optimized Hyperparameters:\n  K={K}, Alpha={ALPHA:.5f}, LR={LR:.6f}, WD={WD:.8f}, Epochs={EPOCHS}")

    for name, s_path, t_path in pairs:
        print(f"\n{'=' * 75}")
        print(f"  DATASET: {name}")
        print(f"{'=' * 75}")

        # Crucially load preprocessed=True to align perfectly with historical data matrices
        print("  Loading data from persistent cached .npy files...")
        try:
            dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
        except Exception as e:
            print(f"  [Error] Could not load preprocessed data: {e}")
            print("  Falling back to preprocessing...")
            dataset = Mydata(s_path, t_path, train=None, preprocessed=False)

        S_data = dataset.S_data if isinstance(dataset.S_data, np.ndarray) else dataset.S_data.numpy()
        T_data = dataset.T_data if isinstance(dataset.T_data, np.ndarray) else dataset.T_data.numpy()
        train_indices = dataset.train_indices
        test_indices = dataset.test_indices

        print(f"  Users: {S_data.shape[0]}, S_items: {S_data.shape[1]}, T_items: {T_data.shape[1]}")
        print(f"  Train users: {len(train_indices)}, Test users: {len(test_indices)}")

        # Run Hybrid Strategy C ('aligned_source') with SLA-Reg and Freezing
        print(f"\n  --- Training Hybrid SLA-CMF ('aligned_source') [SLA-Reg=0.1, Freeze=10] ---")
        hybrid_rmse = train_hybrid_c(
            S_data, T_data, train_indices, test_indices,
            k=K, alpha=ALPHA, lr=LR, wd=WD, epochs=EPOCHS,
            init_mode='aligned_source',
            freeze_u_epochs=10,
            sla_reg_lambda=0.1
        )
        print(f"  >> Hybrid SLA-CMF RMSE: {hybrid_rmse:.4f}")

        # Also run standard CMF with the SAME hyperparams for an instant apple-to-apple delta check
        print(f"\n  --- Running CMF Baseline (Same Hyperparams for delta comparison) ---")
        local_cmf_rmse = train_baseline_cmf(
            S_data, T_data, train_indices, test_indices,
            k=K, alpha=ALPHA, lr=LR, wd=WD, epochs=EPOCHS
        )
        print(f"  >> Local Baseline CMF RMSE: {local_cmf_rmse:.4f}")

        all_results.append({
            'name': name,
            'historical_cmf': historical_cmf.get(name, 0),
            'hybrid': hybrid_rmse,
            'local_cmf': local_cmf_rmse
        })

    # ===============================================================================
    # GENERATE SUMMARY REPORT
    # ===============================================================================
    print(f"\n\n{'=' * 110}")
    print("FINAL COMPARISON: HYBRID SLA-CMF VS BASELINE CMF")
    print(f"{'=' * 110}")
    
    hdr = f"{'Dataset Pair':<40} {'Historical CMF':>16} {'Local CMF':>12} {'Hybrid Model':>14} {'Improv. (%)':>14}"
    print(hdr)
    print("-" * len(hdr))

    with open("final_hybrid_evaluation_results.txt", "w") as f:
        f.write("FINAL COMPARISON: HYBRID SLA-CMF VS BASELINE CMF\n")
        f.write(f"Hyperparameters: K={K}, Alpha={ALPHA:.6f}, LR={LR:.6f}, WD={WD:.8f}\n")
        f.write("=" * 110 + "\n\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")

        for res in all_results:
            hist_cmf = res['historical_cmf']
            loc_cmf = res['local_cmf']
            hyb = res['hybrid']
            
            # Calc improvement vs historical
            improvement = ((hist_cmf - hyb) / hist_cmf) * 100 if hist_cmf > 0 else 0.0
            
            line = f"{res['name']:<40} {hist_cmf:>16.4f} {loc_cmf:>12.4f} {hyb:>14.4f} {improvement:>13.2f}%"
            print(line)
            f.write(line + "\n")

    print(f"\nDetailed final results saved to 'final_hybrid_evaluation_results.txt'")
    print("=" * 110)
