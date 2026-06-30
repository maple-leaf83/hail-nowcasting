"""
mlp_feature_selection.py
------------------------
Recursive feature elimination for the MLP hail nowcasting model using SHAP.

Starting from the full set of 37 predictors, the script iteratively:
    1. Trains a 5-layer MLP on the current feature set.
    2. Computes SHAP values on the test set using the 30-60 min output head.
    3. Ranks features by mean absolute SHAP value and drops the bottom 5%
       (those contributing less than 5% of total cumulative importance).
    4. Repeats until only one feature remains.

BSS (relative to persistence) is recorded at each iteration, allowing the
minimum feature set that retains model skill to be identified.

SHAP is computed on the 30-60 min output head (the hardest lead time) on the
assumption that features important at the longest lead time are a conservative
superset of those needed at shorter lead times.

Currently runs on a single fold (fold 0) for efficiency. Extend range(0, 1)
to range(0, 10) for a full cross-validated feature selection.

Usage:
    python mlp_feature_selection.py -c config.yml

Outputs:
    - mlp_features_selection.csv: BSS and feature count at each iteration.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import yaml
from utils import *
import shutil
from networks import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, roc_auc_score, auc, precision_recall_curve, brier_score_loss
import shap


def format(y):
    """
    Split a multi-output label array into a tuple of per-output arrays,
    as required by Keras multi-output model training.

    Args:
        y (np.ndarray): Array of shape (N, 3) with columns ordered as:
                        [MESH_bool_next_30, MESH_bool_15_45, MESH_bool_last_30]

    Returns:
        Tuple of three 1D arrays (y_next30, y_15_45, y_last30).
    """
    y1 = y[:, 0]  # 0-30 min
    y2 = y[:, 1]  # 15-45 min
    y3 = y[:, 2]  # 30-60 min
    return y1, y2, y3


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog='mlp-training',
                                     description='MLP training script')
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -------------------------------------------------------------------------
    # Load data and derive persistence reference
    # -------------------------------------------------------------------------
    data = pd.read_csv(config['data_file'])
    save_path = config['save_path']
    data['persistence_bool'] = 0
    data.loc[data['MESH_max'] >= 35, 'persistence_bool'] = 1

    # Full predictor set (37 features). Feature selection will progressively
    # reduce this list each iteration based on SHAP importance ranking.
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

    results_df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Outer loop: k-fold cross-validation (currently single fold for speed)
    # -------------------------------------------------------------------------
    for n in range(0, 1):
        print("split = ", n)

        # Extract train/test splits for this fold
        train_df = data.loc[data['split' + str(n)].str.contains('Train'), :]
        val_df = data.loc[data['split' + str(n)].str.contains('Test'), :]
        train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='split')))]
        val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='split')))]
        print("Training df size = ", len(list(train_df)), len(train_df))
        print("Validation df size = ", len(list(val_df)), len(val_df))

        # ---------------------------------------------------------------------
        # Class balancing: undersample majority (no-hail) class so that
        # negatives:positives = mult_factor:1, stratified on MESH_bool_next_30.
        # ---------------------------------------------------------------------
        mult_factor = 3.5
        X_eq_train = train_df.loc[train_df['MESH_bool_next_30'] == 1]
        X_train_0 = train_df.loc[train_df['MESH_bool_next_30'] == 0]
        if int(len(X_eq_train) * mult_factor) >= len(X_train_0):
            X_eq_train = pd.concat([X_eq_train, X_train_0])
        else:
            X_eq_train = pd.concat([X_eq_train, X_train_0.sample(int(len(X_eq_train) * mult_factor))])

        y_train = X_eq_train.loc[:, ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']].values
        X_train = X_eq_train.loc[:, predictors].values

        # Hold-out test set and persistence reference (never used in training)
        y_test = val_df.loc[:, ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']].values
        X_test = val_df.loc[:, predictors]
        y_ref_VP = val_df.loc[:, 'persistence_bool']

        # feature_num tracks the current number of predictors. Starts at the
        # initial count and is updated each iteration by SHAP-based pruning.
        feature_num = len(predictors)

        # -------------------------------------------------------------------------
        # Inner loop: recursive feature elimination
        # Each iteration trains a model, computes SHAP, drops the bottom 5% of
        # features by cumulative importance, and repeats.
        # -------------------------------------------------------------------------
        while feature_num >= 1:
            # Re-split and re-scale each iteration since the feature set changes.
            # Scaler is fit on X_tr only to prevent leakage into X_val and X_test.
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=y_train[:, 0])
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print(y_train.shape, X_train.shape, X_test.shape)

            filters = [[512, 512, 512, 512, 512]]

            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', patience=15, min_lr=1e-10),
                EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
            ]

            model = gen_mlp_network(input_shape=(X_train.shape[1],), num_filters=filters[0], dropout=0.1)
            model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
                          )
            y_train_form = format(y_tr)
            history = model.fit(X_tr, y_train_form, epochs=900, batch_size=512,
                                validation_data=(X_val, format(y_val)),
                                callbacks=callbacks,
                                verbose=0)

            # -----------------------------------------------------------------
            # Evaluate on held-out test set
            # -----------------------------------------------------------------
            test_preds = model.predict(X_test_scaled)
            tp1 = test_preds[0]  # 0-30 min
            tp2 = test_preds[1]  # 15-45 min
            tp3 = test_preds[2]  # 30-60 min
            test_results = np.concatenate([tp1, tp2, tp3], axis=1)

            bs_model1, bs_ref_c1, bs_ref_p1, bss_c1, bss_p1 = compute_BSS_scores(y_test[:, 0], test_results[:, 0], y_ref_VP)
            bs_model2, bs_ref_c2, bs_ref_p2, bss_c2, bss_p2 = compute_BSS_scores(y_test[:, 1], test_results[:, 1], y_ref_VP)
            bs_model3, bs_ref_c3, bs_ref_p3, bss_c3, bss_p3 = compute_BSS_scores(y_test[:, 2], test_results[:, 2], y_ref_VP)

            results_dict = {'split': n, 'num_features': feature_num,
                            'data_factor': mult_factor,
                            'bsm_30': bs_model1, 'BSS_30': bss_p1,
                            'bsm_15': bs_model2, 'BSS_15_45': bss_p2,
                            'bsm_60': bs_model3, 'BSS_30_60': bss_p3}
            print(bss_p1, bss_p2, bss_p3)
            print(results_dict)

            if feature_num == len(predictors) and n == 0:
                results_df = pd.DataFrame(results_dict, index=[0])
            else:
                results_df = pd.concat([results_df, pd.DataFrame(results_dict, index=[len(results_df)])])

            if feature_num > 1:
                # -------------------------------------------------------------
                # SHAP-based feature ranking using the 30-60 min output head.
                # Features are ranked by mean absolute SHAP value. The bottom
                # 5% of cumulative importance are dropped for the next iteration.
                # Using the longest lead time head is conservative — features
                # that matter at 30-60 min are likely relevant at shorter leads.
                # -------------------------------------------------------------
                explainer = shap.Explainer(lambda x: model(x)[2][:, 0],
                                           masker=X_test_scaled,
                                           feature_names=predictors)
                shap_values = explainer(X_test_scaled)

                shap_features = shap_values.feature_names
                shap_abs = np.mean(shap_values.abs.values, axis=0)
                shap_sorted = sorted(shap_abs, reverse=True)
                index = np.argsort(shap_abs)[::-1]
                features_sorted = [shap_features[i] for i in index]

                # Find the index at which cumulative importance reaches 95%
                # of the total. Features beyond this index are dropped.
                shap_cumsum = np.cumsum(shap_sorted)
                feature_index = [i for i, s in enumerate(shap_cumsum) if s <= 0.95 * np.max(shap_cumsum)]

                feature_num = feature_index[-1]
                print("New # features = ", feature_num)
                predictors_updated = features_sorted[0:feature_num]
                print("predictors_updated = ", predictors_updated)

                # Update feature set and re-extract arrays for next iteration
                predictors = predictors_updated
                X_train = X_eq_train.loc[:, predictors].values
                X_test = val_df.loc[:, predictors].values
            else:
                break

    results_df.to_csv('mlp_features_selection.csv', index=False)
