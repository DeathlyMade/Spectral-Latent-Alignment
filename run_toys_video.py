import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from Data_Preprocessing import Mydata
from CMF import train_cmf
from SLA import train_sla

s_path = r"./Data/ratings_Toys_and_Games.csv"
t_path = r"./Data/ratings_Video_Games.csv"

print("Regenerating datasets to fix dimensions...")
dataset = Mydata(s_path, t_path, train=None, preprocessed=False)
print("Dataset created.")

# Because the script saves to .npy, let's load it as preprocessed to match standard flow
dataset_proc = Mydata(s_path, t_path, train=None, preprocessed=True)
print("S_data shape:", dataset_proc.S_data.shape)
print("T_data shape:", dataset_proc.T_data.shape)

print("\n--- Running CMF ---")
# Using best params from CMF.py or defaults
rmse_cmf = train_cmf(s_path, t_path, k=50, alpha=0.5, lr=0.01, wd=1e-3, epochs=30)
print(f"CMF RMSE: {rmse_cmf:.4f}")

print("\n--- Running SLA ---")
S_data = dataset_proc.S_data
T_data = dataset_proc.T_data
train_indices = dataset_proc.train_indices
test_indices = dataset_proc.test_indices

rmse_sla = train_sla(S_data, T_data, train_indices, test_indices, k=100, epochs=50, lr=0.005)
print(f"SLA RMSE: {rmse_sla:.4f}")
