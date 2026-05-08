# Hyperparameter Tuning Plan for SLA

The current baseline to beat for the Spectral Latent Alignment (SLA) model on the `Office Products -> Movies and TV` dataset pair is the CMF model, which achieves an RMSE of `0.9939`. Currently, SLA's best out-of-the-box configuration is around `1.09`. To bridge this gap and beat CMF, we will perform an extensive hyperparameter search using `optuna`.

## Proposed Changes

### Optuna Hyperparameter Tuning Script (`run_sla_optuna.py`)

We will create a new Python script that leverages `optuna` for Bayesian optimization of the SLA model. To make the search efficient, we will separate the heavy spectral embedding computation from the predictive model training loop.

#### 1. Optimization Strategy (Precomputation)
Computing the Singular Value Decomposition (SVD) for spectral embeddings takes a significant chunk of time. Since the SVD only depends on the interaction matrix and `k`, we will precompute and cache the embeddings and Procrustes alignments for a predefined grid of `k` values.

#### 2. Search Space
The Optuna objective will select from these precomputed embeddings and then tune the predictive model.

**SLA Specifics:**
*   **`k` (Spectral Dimension)**: `[50, 100, 150, 200, 250, 300, 400, 500]`
*   **`use_ratings`**: `False` (Binary Adjacency has proven much better empirically, but we can allow `[True, False]`)

**Predictive Model (DeepMLP):**
*   **Layers**: 1 to 4 hidden layers
*   **Hidden Units**: `[32, 64, 128, 256, 512]`
*   **Dropout**: `0.0` to `0.6`
*   **Optimizer**: `Adam`, `AdamW`, `RMSprop`
*   **Learning Rate**: Log-uniform `1e-4` to `1e-2`
*   **Weight Decay**: Log-uniform `1e-6` to `1e-2`
*   **Batch Size**: `[128, 256, 512, 1024]`
*   **Epochs**: Fixed at a higher number (e.g., 100-200) with Early Stopping on a validation split to prevent overfitting.

**Alternative Models:**
We will also allow Optuna to select tree-based models (e.g., `HistGradientBoostingRegressor`, `XGBoost`, `RandomForest`, `Ridge`) to see if an ensemble method outperforms the MLP on the aligned embeddings.

### Estimated Execution Time

*   **Precomputation Phase**: Computing the embeddings for 8 `k` values takes ~1-2 minutes.
*   **Trial Execution**: Training a DeepMLP with early stopping or a tree model takes roughly 5 to 15 seconds per trial.
*   **Total Tuning Time**: For `200` Optuna trials, the entire hyperparameter tuning loop will take approximately **20 to 45 minutes** depending on the complexity of the sampled models.

## User Review Required

> [!IMPORTANT]
> - Do you want the script to automatically run the 200 Optuna trials immediately after creating the script, or would you prefer to run it manually later? 
> - If we run it now, we will wait for the ~30 minutes execution to see if we beat CMF. Alternatively, we could run a shorter grid (e.g., 50 trials) for a quicker feedback loop. How many trials would you prefer?
> - Are there any specific datasets other than `Office Products -> Movies and TV` you'd like to prioritize for the tuning?
