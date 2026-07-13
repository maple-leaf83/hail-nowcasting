"""
xgb_feature_selection.py
------------------------
Recursive feature elimination for the XGBoost hail nowcasting models using SHAP.

A separate feature selection run is performed for each of the three lead times
(0-30, 15-45, 30-60 min). Starting from the full set of 37 predictors, the
script iteratively:
    1. Trains an XGBoost model on the current feature set.
    2. Computes SHAP values on the test set.
    3. Ranks features by mean absolute SHAP value and drops the bottom 5%
       by count (at least 1 per iteration).
    4. Repeats until only one feature remains.

BSS (relative to persistence) and the active feature set are recorded at every
iteration, including the single-feature case.

Hyperparameters are loaded from pre-saved JSON files (produced by xgboost_training.py).
Currently runs on fold 0 only for efficiency.

Usage:
    python xgb_feature_selection.py -c config.yml

Outputs:
    - xgb_features_next30.csv   : BSS and feature set at each iteration (0-30 min)
    - xgb_features_15.csv       : BSS and feature set at each iteration (15-45 min)
    - xgb_features_last30.csv   : BSS and feature set at each iteration (30-60 min)
"""

import os
import pandas as pd
import json
import shap
import argparse
import yaml
from utils import *

def get_best_parameters(path_first_30, path_15_45, past_last30, train_df, predictors, predictands):
    if os.path.exists(path_first_30) and os.path.exists(path_15_45) and os.path.exists(past_last30):
        print("Files found. Loading best parameters from JSON files.")
        # Load the best parameters from the file
        with open(path_first_30, 'r') as f:
            best_params_next_30 = json.load(f)

        with open(past_last30, 'r') as f:
            best_params_last_30 = json.load(f)

        with open(path_15_45, 'r') as f:
            best_params_15_45 = json.load(f)
        print("Loaded best parameters from JSON files.")
    else:
        print("Files not found. Performing GridSearchCV to find the best parameters.")
        """ perform grid search and save parameters """
        y_train = train_df.loc[:, predictands[0]].values
        X_train = train_df.loc[:, predictors].values
        # Perform GridSearchCV to find the best parameters
        best_params_next_30 = perform_grid_search(predictands[0], X_train, y_train)
        # Save the best parameters to a file
        with open(path_first_30, 'w') as f:
            json.dump(best_params_next_30, f)

        y_train = train_df.loc[:, predictands[2]].values
        best_params_last_30 = perform_grid_search(predictands[2], X_train, y_train)
        with open(past_last30, 'w') as f:
            json.dump(best_params_last_30, f)

        y_train = train_df.loc[:, predictands[1]].values
        best_params_15_45 = perform_grid_search(predictands[1], X_train, y_train)
        with open(path_15_45, 'w') as f:
            json.dump(best_params_15_45, f)

    return best_params_next_30, best_params_15_45, best_params_last_30


def feature_selection(train_df, val_df, predictand, predictors, params, n):
    """
    Recursive feature elimination for a single XGBoost model using SHAP.

    Starting from the full predictor set, iteratively:
        1. Trains an XGBoost model on the current feature set.
        2. Computes SHAP values on the test set.
        3. Ranks features by mean absolute SHAP value and drops the bottom 5%
           by count (at least 1 feature per iteration).
        4. Repeats until only one feature remains.

    Brier score, climatology reference score, and BSS are recorded at every
    iteration including the single-feature case.

    Args:
        train_df (pd.DataFrame): Training partition for the current fold.
        val_df (pd.DataFrame):   Test partition for the current fold.
        predictand (str):        Target column name.
        predictors (list):       Initial list of predictor feature names (copied internally).
        params (dict):           XGBoost hyperparameters.
        n (int):                 Fold index (recorded in output for traceability).

    Returns:
        list of dict: One result dict per iteration, containing split, predictand,
                      num_features, features, bsm, bsref, and bss.
    """
    y_train = train_df.loc[:, predictand].values
    y_test = val_df.loc[:, predictand].values

    # Initialise feature count from the actual predictor list (not hardcoded)
    feature_num = len(predictors)
    X_train = train_df.loc[:, predictors].values
    X_test = val_df.loc[:, predictors].values

    all_results = []

    while feature_num >= 1:
        print("training/testing data shape:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        results, xgb_model = train_for_feature_selection(X_train, y_train, X_test, y_test, predictors,
                                                         params, smote_flag=0)

        # Collect all metrics — bsm and bsref were previously only printed
        all_results.append({
            'split':        n,
            'predictand':   predictand,
            'num_features': feature_num,
            'features':     ','.join(predictors),
            'bsm':          results['bsm'],
            'bsref':        results['bsref'],
            'bss':          results['bss'],
        })
        print(f"  predictand={predictand}, n_features={feature_num}, "
              f"bsm={results['bsm']:.4f}, bsref={results['bsref']:.4f}, bss={results['bss']:.4f}")

        if feature_num == 1:
            break

        # ---------------------------------------------------------------------
        # SHAP-based feature ranking. Drop the bottom 5% of features by count
        # each iteration. max(1, ...) ensures at least one feature is always
        # dropped so the loop always terminates.
        # ---------------------------------------------------------------------
        X_test_df = val_df.loc[:, predictors]
        explainer = shap.TreeExplainer(xgb_model, X_test_df)
        shap_values = explainer(X_test_df, check_additivity=False)

        shap_abs = np.mean(shap_values.abs.values, axis=0)
        index = np.argsort(shap_abs)[::-1]
        features_sorted = [predictors[i] for i in index]

        n_drop = max(1, int(np.floor(feature_num * 0.05)))
        feature_num = feature_num - n_drop
        predictors = features_sorted[:feature_num]
        print(f"  Dropped {n_drop} feature(s). Remaining ({feature_num}): {predictors}")

        # Update training and testing arrays for next iteration
        X_train = train_df.loc[:, predictors].values
        X_test = val_df.loc[:, predictors].values

    return all_results



if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog='xgb-feature-selection',
                                     description='Recursive SHAP-based feature selection for XGBoost')
    parser.add_argument("-c", "--config", help="Path to config YAML file")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -------------------------------------------------------------------------
    # Load data and extract fold 0 train/test split
    # -------------------------------------------------------------------------
    data = pd.read_csv(config['data_file'])

    predictands = ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']
    save_path = 'params/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # -------------------------------------------------------------------------
    # Predictor features (37 total)
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

    # Run on fold 0 only for efficiency (extend range to run across all folds)
    n = 0
    print("split = ", n)
    train_df = data.loc[data['split' + str(n)].str.contains('Train'), :]
    val_df = data.loc[data['split' + str(n)].str.contains('Test'), :]
    train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='split')))]
    val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='split')))]
    print("Training df size = ", len(list(train_df)), len(train_df))
    print("Validation df size = ", len(list(val_df)), len(val_df))

    # -------------------------------------------------------------------------
    # Load pre-computed hyperparameters and run feature selection per lead time.
    # Hyperparameters are loaded from JSON files saved by xgboost_training.py.
    # A fresh copy of the full predictor list is passed to each call so that
    # feature selection for each lead time starts independently from all 37
    # features. Results from all three are collected and written to one CSV.
    # -------------------------------------------------------------------------
    all_results = []

    print("\n--- 0-30 min ---")
    next_30_params_path = os.path.join(save_path, 'best_params_next_30_split0.json')
    with open(next_30_params_path, 'r') as f:
        best_params_next_30 = json.load(f)
    all_results += feature_selection(train_df, val_df, 'MESH_bool_next_30', list(predictors),
                                     best_params_next_30, n)

    print("\n--- 15-45 min ---")
    params_path_15_45 = os.path.join(save_path, 'best_params_15_45_split0.json')
    with open(params_path_15_45, 'r') as f:
        best_params_15_45 = json.load(f)
    all_results += feature_selection(train_df, val_df, 'MESH_bool_15_45', list(predictors),
                                     best_params_15_45, n)

    print("\n--- 30-60 min ---")
    last_30_params_path = os.path.join(save_path, 'best_params_last_30_split0.json')
    with open(last_30_params_path, 'r') as f:
        best_params_last_30 = json.load(f)
    all_results += feature_selection(train_df, val_df, 'MESH_bool_last_30', list(predictors),
                                     best_params_last_30, n)

    # Write all results to a single CSV
    pd.DataFrame(all_results).to_csv('xgb_feature_selection.csv', index=False)
    print("\nResults saved to xgb_feature_selection.csv")

