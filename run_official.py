import os
import subprocess
import sys

dataset_pairs = [
    (r"d:\Recsys\ratings_Office_Products.csv", r"d:\Recsys\ratings_Movies_and_TV.csv", "Office Products -> Movies and TV"),
    (r"d:\Recsys\ratings_Sports_and_Outdoors.csv", r"d:\Recsys\ratings_CDs_and_Vinyl.csv", "Sports and Outdoors -> CDs and Vinyls"),
    (r"d:\Recsys\ratings_Apps_for_Android.csv", r"d:\Recsys\ratings_Video_Games.csv", "Android Apps -> Video Games"),
    (r"d:\Recsys\ratings_Toys_and_Games.csv", r"d:\Recsys\ratings_Automotive.csv", "Toys and Games -> Automotive")
]

venv_python = r"d:\Recsys\venv\Scripts\python.exe"
results_file = r"d:\Recsys\official_results.txt"

with open(results_file, "w") as f:
    f.write(f"{'Dataset Pair':<40} | {'U-DARec RMSE':<15} | {'I-DARec RMSE':<15} | {'DARec RMSE':<15}\n")
    f.write("-" * 95 + "\n")

# Import original DARec runner dependencies
sys.path.append(r"d:\Recsys")
from Data_Preprocessing import Mydata
from run_darec import run_darec_on_data

for s_path, t_path, name in dataset_pairs:
    print(f"\n==========================================")
    print(f"Running experiments for: {name}")
    print(f"==========================================")

    u_rmse = "N/A"
    i_rmse = "N/A"
    darec_rmse_str = "N/A"

    # 1. Run U-DARec
    print("\n>>> Running U-DARec AutoRec (Source)...")
    subprocess.run([venv_python, r"d:\Recsys\U-DARec\Train_AutoRec.py", "--s_path", s_path, "--t_path", t_path, "--train_S", "1"], check=True)
    print(">>> Running U-DARec AutoRec (Target)...")
    subprocess.run([venv_python, r"d:\Recsys\U-DARec\Train_AutoRec.py", "--s_path", s_path, "--t_path", t_path, "--train_S", "0"], check=True)
    print(">>> Running U-DARec...")
    res_u = subprocess.run([venv_python, r"d:\Recsys\U-DARec\Train_DArec.py", "--s_path", s_path, "--t_path", t_path], capture_output=True, text=True)
    print(res_u.stdout[-200:])
    for line in res_u.stdout.split('\n'):
        if "Min test RMSE:" in line:
            u_rmse = f"{float(line.split(':')[-1].strip()):.4f}"

    # 2. Run I-DARec
    print("\n>>> Running I-DARec AutoRec (Source)...")
    subprocess.run([venv_python, r"d:\Recsys\I-DARec\Train_AutoRec.py", "--s_path", s_path, "--t_path", t_path, "--train_S", "1"], check=True)
    print(">>> Running I-DARec AutoRec (Target)...")
    subprocess.run([venv_python, r"d:\Recsys\I-DARec\Train_AutoRec.py", "--s_path", s_path, "--t_path", t_path, "--train_S", "0"], check=True)
    print(">>> Running I-DARec...")
    res_i = subprocess.run([venv_python, r"d:\Recsys\I-DARec\Train_DArec.py", "--s_path", s_path, "--t_path", t_path], capture_output=True, text=True)
    print(res_i.stdout[-200:])
    for line in res_i.stdout.split('\n'):
        if "Min test RMSE:" in line:
            i_rmse = f"{float(line.split(':')[-1].strip()):.4f}"

    # 3. Run original DARec
    print("\n>>> Running DARec...")
    try:
        dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
        darec_rmse = run_darec_on_data(dataset.S_data, dataset.T_data, dataset.train_indices, dataset.test_indices, epochs=1500)
        darec_rmse_str = f"{darec_rmse:.4f}"
    except Exception as e:
        print(f"Error running DARec: {e}")
        darec_rmse_str = "Error"

    print(f"Results for {name}: U-DARec = {u_rmse}, I-DARec = {i_rmse}, DARec = {darec_rmse_str}")
    with open(results_file, "a") as f:
        f.write(f"{name:<40} | {u_rmse:<15} | {i_rmse:<15} | {darec_rmse_str:<15}\n")
    
print("\nAll experiments completed. Results saved to official_results.txt.")
