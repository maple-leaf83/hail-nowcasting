"""
mlp_shap_analysis.py
--------------------
Computes SHAP (SHapley Additive exPlanations) values for a trained MLP hail
nowcasting model, for a single cross-validation fold.

For each of the three forecast lead times (0-30, 15-45, 30-60 min), a separate
shap.Explainer is constructed using the corresponding model output head. SHAP
values are computed on the held-out test set for that fold and saved to disk as
pickle files for downstream figure generation.

The script is designed to be run once per fold, with the fold index supplied via
the --split argument. This makes it straightforward to parallelise across folds
on a cluster (e.g. one job per fold).

Note on StandardScaler: The scaler is fit on the full training partition of the
fold (not the undersampled set), so that the test set is scaled relative to the
true training distribution.

Usage:
    python mlp_shap_analysis.py -c config.yml -s <fold_index>

    <fold_index>: integer in [0, 9] corresponding to the cross-validation fold.

Outputs (one file per lead time per fold):
    shap_values_split_<n>_MESH_bool_next_30.pkl   (0-30 min)
    shap_values_split_<n>_MESH_bool_15_45.pkl     (15-45 min)
    shap_values_split_<n>_MESH_bool_last_30.pkl   (30-60 min)

    Each pickle file contains a shap.Explanation object with:
        .values          : array of shape (N_test, N_features)
        .base_values     : model base value (expected output)
        .feature_names   : list of predictor names
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import yaml
from utils import *
import shutil
from networks import *
import shap
import pickle

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration: parse command-line arguments and load config YAML
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog='mlp-shap',
                                     description='Compute SHAP values for one fold of the MLP nowcasting model')
    parser.add_argument("-c", "--config", help="Path to config YAML file")
    parser.add_argument("-s", "--split", help="Fold index (0-9) for which to compute SHAP values")
    args = parser.parse_args()
    config_file = args.config
    print(args)
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data = pd.read_csv(config['data_filename'])

    # -------------------------------------------------------------------------
    # Predictor features (37 total)
    # Includes radar-derived storm properties, lightning, and reanalysis-derived
    # environmental parameters. See README for full descriptions.
    # -------------------------------------------------------------------------
    predictors = ['MESH_area_ge_20', 'MESH_max', 'MESH_p90', 'storm_area', 'MUCAPE', 'MUCAPEm10m30',
                  'storm_duration', 'MUEL', 'ETH_p90', 'MUCIN', 'storm_area_trend', 'lightning_flash_rate',
                  'WBFZL', 'MESH_p90_trend', 'ETH_p90_trend', 'RH36mean', 'storm_motion_x',
                  'SRH03r', 'azshear_p90', 'U06mean', 'storm_motion_mag', 'SRH03l', 'MUVTEm20',
                  'lightning_flash_rate_trend', 'MESH_area_ge_20_trend', 'MESH_max_trend', 'deviant_motion_mag',
                  'storm_motion_y', 'V06mean', 'PW', 'deviant_motion_x', 'azshear_p90_trend',
                  'BWD06',
                  'deviant_motion_y', 'lightning_flash_rate_density_trend',
                  'lightning_flash_rate_density', 'LR36'
                  ]
    save_path = 'params/'

    # -------------------------------------------------------------------------
    # Extract train/test split for the specified fold
    # -------------------------------------------------------------------------
    n = int(args.split)
    print("split = ", args.split)
    train_df = data.loc[data['split' + str(n)].str.contains('Train'), :]
    val_df = data.loc[data['split' + str(n)].str.contains('Test'), :]
    # Drop all split indicator columns from both sets
    train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='split')))]
    val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='split')))]

    # -------------------------------------------------------------------------
    # Feature scaling
    # Scaler is fit on the full (unsampled) training partition to ensure the
    # test set is scaled relative to the true training distribution.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_tr = train_df.loc[:, predictors].values
    X_test = val_df.loc[:, predictors].values
    X_tr = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_test)
    print(X_tr.shape, X_test_scaled.shape)
    print("Training df size = ", len(list(train_df)), len(train_df))
    print("Validation df size = ", len(list(val_df)), len(val_df))

    # -------------------------------------------------------------------------
    # Load trained model weights for this fold
    # Architecture must match what was used during training (see networks.py).
    # -------------------------------------------------------------------------
    model_path = os.path.join(save_path, 'nowcasting_model' + str(n) + '.keras')
    model = gen_mlp_network(input_shape=(X_test_scaled.shape[1],), num_filters=[512, 512, 512, 512, 512], dropout=0.1)
    model.load_weights(model_path)

    # -------------------------------------------------------------------------
    # Compute SHAP values for each output head (one per lead time)
    # A separate Explainer is required for each output because shap.Explainer
    # wraps a single scalar-output function. The lambda selects the i-th output
    # head and extracts the scalar probability (shape N -> scalar).
    # SHAP values are saved as pickle files for use by gen_mean_shap.py and
    # gen_shap_figs_ultraplot.py.
    # -------------------------------------------------------------------------
    targets = ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']
    shap_values = []
    for i in range(len(model.outputs)):
        print(f"Calculating SHAP values for output {i + 1} ({targets[i]})")
        explainer = shap.Explainer(lambda x: model(x)[i][:, 0],
                                   masker=X_test_scaled,
                                   feature_names=predictors)
        shap_values = explainer(X_test_scaled)

        fname = 'shap_values_split_' + str(n) + '_' + targets[i] + '.pkl'
        print(f"Saving SHAP values to {fname}")
        with open(fname, "wb") as f:
            pickle.dump(shap_values, f)
