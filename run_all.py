import os
from Data_Preprocessing import Mydata
from SLA import train_sla
from run_darec import run_darec_on_data

# Disable TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pairs = [
    ("Office Products -> Movies and TV", 'ratings_Office_Products.csv', 'ratings_Movies_and_TV.csv'),
    ("Sports and Outdoors -> CDs and Vinyls", 'ratings_Sports_and_Outdoors.csv', 'ratings_CDs_and_Vinyl.csv'),
    ("Android Apps -> Video Games", 'ratings_Apps_for_Android.csv', 'ratings_Video_Games.csv'),
    ("Toys and Games -> Automotive", 'ratings_Toys_and_Games.csv', 'ratings_Automotive.csv')
]

base_path = r'D:\Recsys'

results = []

for name, s_file, t_file in pairs:
    print("=" * 60)
    print(f"Running experiments for: {name}")
    s_path = os.path.join(base_path, s_file)
    t_path = os.path.join(base_path, t_file)
    
    # Load dataset.
    # Note: If memory becomes an issue, we can rely on saved .npy
    # If the user has already run the preprocessed=False in the past, .npy files might exist.
    # Let's set preprocessed=False to ensure the script parses the CSV files if not already done.
    # But wait! The .npy files already exist for all these CSV files in the dir. 
    # list_dir showed .csv.npy files. Let's use preprocessed=True to speed up!
    
    print(f"Loading data from {s_file} and {t_file}...")
    try:
        # Try loading preprocessed
        dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
    except Exception as e:
        print("Preprocessed files not found or corrupted, generating them from CSV...")
        dataset = Mydata(s_path, t_path, train=None, preprocessed=False)

    S_data = dataset.S_data
    T_data = dataset.T_data
    train_indices = dataset.train_indices
    test_indices = dataset.test_indices
    
    print("Running SLA_Recsys...")
    rmse_sla = train_sla(S_data, T_data, train_indices, test_indices, k=100, epochs=20)
    
    print("Running DARec...")
    rmse_darec = run_darec_on_data(S_data, T_data, train_indices, test_indices, epochs=50)
    
    print(f"Results for {name}: SLA_RMSE = {rmse_sla:.4f}, DARec_RMSE = {rmse_darec:.4f}")
    results.append((name, rmse_sla, rmse_darec))

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"{'Dataset Pair':<40} | {'SLA RMSE':<10} | {'DARec RMSE':<10}")
print("-" * 65)
with open("d:/Recsys/results.txt", "w") as f:
    f.write("FINAL RESULTS\n")
    for name, r_sla, r_darec in results:
        line = f"{name:<40} | {r_sla:<10.4f} | {r_darec:<10.4f}"
        print(line)
        f.write(line + "\n")
