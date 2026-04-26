import subprocess
import sys

venv_python = r"d:\Recsys\venv\Scripts\python.exe"

dataset_pairs = [
    (r"d:\Recsys\ratings_Office_Products.csv", r"d:\Recsys\ratings_Movies_and_TV.csv", "Office Products -> Movies and TV"),
    (r"d:\Recsys\ratings_Sports_and_Outdoors.csv", r"d:\Recsys\ratings_CDs_and_Vinyl.csv", "Sports and Outdoors -> CDs and Vinyls"),
    (r"d:\Recsys\ratings_Apps_for_Android.csv", r"d:\Recsys\ratings_Video_Games.csv", "Android Apps -> Video Games"),
    (r"d:\Recsys\ratings_Toys_and_Games.csv", r"d:\Recsys\ratings_Automotive.csv", "Toys and Games -> Automotive")
]

for s_path, t_path, name in dataset_pairs:
    print(f"\n==========================================")
    print(f"Running True I-DARec for: {name}")
    print(f"==========================================")
    
    print(">>> Running I-DARec AutoRec (Source)...")
    subprocess.run([venv_python, r"d:\Recsys\I-DARec\Train_AutoRec.py", "--s_path", s_path, "--t_path", t_path, "--train_S", "1"], check=True)
    
    print(">>> Running I-DARec AutoRec (Target)...")
    subprocess.run([venv_python, r"d:\Recsys\I-DARec\Train_AutoRec.py", "--s_path", s_path, "--t_path", t_path, "--train_S", "0"], check=True)
    
    print(">>> Running I-DARec...")
    res_i = subprocess.run([venv_python, r"d:\Recsys\I-DARec\Train_DArec.py", "--s_path", s_path, "--t_path", t_path], capture_output=True, text=True)
    
    i_rmse = "Error"
    for line in res_i.stdout.split('\n'):
        if "Min test RMSE:" in line:
            i_rmse = f"{float(line.split(':')[-1].strip()):.4f}"
            
    print(f"True I-DARec RMSE for {name}: {i_rmse}")
    with open("true_idarec_results.txt", "a") as f:
        f.write(f"{name}: {i_rmse}\n")
        
print("Finished evaluating true I-DARec metrics.")
