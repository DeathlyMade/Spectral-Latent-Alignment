import sys, os
import numpy as np
from scipy.sparse.linalg import svds

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.append(os.path.join(os.path.dirname(__file__), 'darec'))
from darec.DARec import DARec

def get_embeddings(R, k=500):
    from scipy.sparse import csr_matrix
    R_sparse = csr_matrix(R)
    actual_k = min(k, min(R.shape) - 1)
    u, s, vt = svds(R_sparse, k=actual_k)
    return u

def run_darec_on_data(S_data, T_data, train_indices, test_indices, epochs=1500):
    # S_data, T_data are dense numpy matrices
    # S_data: (num_users, num_items_S)
    # T_data: (num_users, num_items_T)
    
    train_matrix_sc = np.zeros_like(S_data)
    train_matrix_sc[train_indices] = S_data[train_indices]
    
    test_matrix_sc = np.zeros_like(S_data)
    test_matrix_sc[test_indices] = S_data[test_indices]

    train_matrix_tg = np.zeros_like(T_data)
    train_matrix_tg[train_indices] = T_data[train_indices]
    
    test_matrix_tg = np.zeros_like(T_data)
    test_matrix_tg[test_indices] = T_data[test_indices]

    # DARec uses original_matrix_sc to get shapes
    original_matrix_sc = S_data
    original_matrix_tg = T_data

    embedding_arr_sc = get_embeddings(S_data, k=500)
    embedding_arr_tg = get_embeddings(T_data, k=500)

    # We need a fresh graph for each run
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        model = DARec(
            sess,
            training_mode='dann',
            input_dim=embedding_arr_sc.shape[1],
            pred_dim=60,
            shared_dim=32,
            pred_sc_tg_lambda=0.1,    
            pred_reg=1e-6,
            cls_layers=[16,8,2],
            cls_reg=1e-5,
            grl_lambda=0.1,
            drop_out_rate=0.25,
            dec_nn_dim_sc=500,
            dec_nn_dim_tg=500,
            domain_loss_ratio= 1,
            mode='user',
            lr=0.001,
            epochs=epochs,
            batch_size=128,
            T=10 ** 3,
            verbose=False
        )

        model.prepare_data(original_matrix_sc, train_matrix_sc, test_matrix_sc,
                           original_matrix_tg, train_matrix_tg, test_matrix_tg,
                           embedding_arr_sc, embedding_arr_tg)

        model.build_model()
        model.train()
        
        mae, rmse = model.eval_one_epoch(-1)
        return rmse
