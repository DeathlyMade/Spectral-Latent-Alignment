"""
Optuna Hyperparameter Tuner for Hybrid SLA-CMF (Strategy C)
============================================================
Optimizes embedding dimension (k), domain weight (alpha), learning rate (lr), 
and weight decay (wd) on the "Office Products -> Movies and TV" pair.
"""

import os
import sys
import warnings
import numpy as np
import optuna

warnings.filterwarnings('ignore')

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Data_Preprocessing import Mydata
from HybridSLACMF import train_hybrid_c

# Global Dataset Loader
def load_dataset():
    s_path = r"./Data/ratings_Office_Products.csv"
    t_path = r"./Data/ratings_Movies_and_TV.csv"
    
    print("  Loading data (regenerating for tuning)...")
    dataset = Mydata(s_path, t_path, train=None, preprocessed=False)
    
    S_data = dataset.S_data if isinstance(dataset.S_data, np.ndarray) else dataset.S_data.numpy()
    T_data = dataset.T_data if isinstance(dataset.T_data, np.ndarray) else dataset.T_data.numpy()
    train_indices = dataset.train_indices
    test_indices = dataset.test_indices
    
    return S_data, T_data, train_indices, test_indices

# Load data once globally
S_data, T_data, train_indices, test_indices = load_dataset()

def objective(trial):
    # Suggest hyperparameters
    k = trial.suggest_categorical('k', [25, 50, 75, 100, 150])
    alpha = trial.suggest_float('alpha', 0.1, 0.9)
    lr = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    
    # Run the hybrid model (aligned_source mode only)
    rmse = train_hybrid_c(
        S_data, T_data, train_indices, test_indices,
        k=k, alpha=alpha, lr=lr, wd=wd, epochs=45,
        init_mode='aligned_source'
    )
    
    return rmse

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING HYBRID SLA-CMF HYPERPARAMETER TUNING (50 TRIALS)")
    print("=" * 80)
    
    # Suppress verbose trial outputs from optuna to keep terminal clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="minimize", study_name="hybrid_sla_cmf_opt")
    
    # Callback to print clean trial logs
    def logging_callback(study, trial):
        print(f"  Trial {trial.number:02d}/50 | RMSE: {trial.value:.4f} "
              f"| Best RMSE: {study.best_value:.4f} "
              f"| params: k={trial.params['k']}, alpha={trial.params['alpha']:.3f}, "
              f"lr={trial.params['lr']:.4f}, wd={trial.params['wd']:.6f}")
              
    study.optimize(objective, n_trials=20, callbacks=[logging_callback])
    
    # Print best results
    print("\n" + "=" * 80)
    print("TUNING COMPLETED!")
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Save best parameters to file
    with open("hybrid_best_params.txt", "w") as f:
        f.write("BEST HYPERPARAMETERS FOR HYBRID SLA-CMF (Strategy C)\n")
        f.write(f"Dataset Pair: Office Products -> Movies and TV\n")
        f.write(f"Best RMSE: {study.best_value:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Optimal Parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTop 5 Trials:\n")
        trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        for i, t in enumerate(trials_sorted[:5]):
            f.write(f"  #{i+1}: RMSE={t.value:.4f} | {t.params}\n")
            
    print("Best params saved to hybrid_best_params.txt\n")
