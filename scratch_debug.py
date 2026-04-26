from Data_Preprocessing import Mydata
from run_darec import run_darec_on_data
import sys
import traceback

try:
    s_path = r"d:\Recsys\ratings_Office_Products.csv"
    t_path = r"d:\Recsys\ratings_Movies_and_TV.csv"
    dataset = Mydata(s_path, t_path, train=True, preprocessed=True)
    darec_rmse = run_darec_on_data(dataset.S_data, dataset.T_data, dataset.train_indices, dataset.test_indices, epochs=1)
    print(f"Success! RMSE: {darec_rmse}")
except Exception as e:
    print("Error:")
    traceback.print_exc()
