"""
xgboost_training.py
--------------------
Trains three XGBoost classifiers to predict the probability of severe hail
(MESH >= 35 mm) at three forecast lead times:
    - 0-30 minutes  (MESH_bool_next_30)
    - 15-45 minutes (MESH_bool_15_45)
    - 30-60 minutes (MESH_bool_last_30)

A separate model is trained per lead time per fold, as the relationship between
predictors and hail occurrence differs across forecast horizons.

Hyperparameters are loaded from pre-saved JSON files if available (from a prior
GridSearchCV run), otherwise GridSearchCV is performed and results saved. This
avoids expensive re-tuning on every training run.

Models are evaluated using the Brier Skill Score (BSS) relative to both
climatology and a persistence forecast.

Usage:
    python xgboost_training.py -c config.yml

Outputs:
    - Trained models:      params/xgboost_<lead>_<fold>.json  (one per lead time per fold)
    - Feature importances: params/xgb_features_<lead>_<fold>.csv
    - Skill score results: nowcasting_XGB_results.csv
    - Full predictions:    xgb_predictions.csv
"""

import os
import pandas as pd

import json
# import shap
import logging

import argparse
import yaml
from utils import *
import shutil



def gen_kfolds_based_on_days(df, K):
    """
    Generate K-fold cross-validation splits stratified by day.

    Days are shuffled randomly (seed=42) and divided into K contiguous blocks.
    Splitting by day rather than by sample prevents temporal leakage, since
    storm observations within the same day are not independent.

    Args:
        df (pd.DataFrame): Input dataframe with a 'time_unix' column.
        K (int): Number of folds.

    Returns:
        pd.DataFrame: Input dataframe with K additional columns 'split0'...'splitK-1',
                      each containing 'Train' or 'Test'.

    Note: Called once to pre-compute splits and save them to the training CSV.
    Not called during normal training runs.
    """
    unique_days = df.loc[:,'time_unix'].unique()
    print(len(unique_days))
    # Shuffle the days
    np.random.seed(42)
    np.random.shuffle(unique_days)

    for n in range(K):
        df['split'+str(n)] =""
        lower_lim = int(float(n*len(unique_days))/K)
        upper_lim = int(float((n+1)*len(unique_days))/K)

        if n == K-1:
            upper_lim = len(unique_days)
        test_days = unique_days[lower_lim:upper_lim]
        df.loc[df['time_unix'].isin(test_days), 'split'+str(n)] = 'Test'
        df.loc[~df['time_unix'].isin(test_days), 'split' + str(n)] = 'Train'

    return df

def get_best_parameters(path_first_30, path_15_45, past_last30, train_df, predictors, predictands):
    """
    Load or compute XGBoost hyperparameters for all three lead times.

    If JSON files exist at all three paths, loads parameters from disk.
    Otherwise runs GridSearchCV (via perform_grid_search in utils.py) for each
    lead time and saves the results. This avoids re-running expensive hyperparameter
    search on every training run.

    Note: Currently the same split-0 parameters are reused across all folds.
    Per-fold tuning could improve performance but at significant compute cost.

    Args:
        path_first_30 (str): Path to JSON file for 0-30 min parameters.
        path_15_45 (str):    Path to JSON file for 15-45 min parameters.
        past_last30 (str):   Path to JSON file for 30-60 min parameters.
        train_df (pd.DataFrame): Training data for the current fold (used only
                                 if grid search is needed).
        predictors (list):   Feature names.
        predictands (list):  Target column names, ordered as
                             [next_30, 15_45, last_30].

    Returns:
        best_params_next_30 (dict): Hyperparameters for 0-30 min model.
        best_params_15_45 (dict):   Hyperparameters for 15-45 min model.
        best_params_last_30 (dict): Hyperparameters for 30-60 min model.
    """
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
        with open(next_30_params_path, 'w') as f:
            json.dump(best_params_next_30, f)

        y_train = train_df.loc[:, predictands[2]].values
        best_params_last_30 = perform_grid_search(predictands[2], X_train, y_train)
        with open(last_30_params_path, 'w') as f:
            json.dump(best_params_last_30, f)

        y_train = train_df.loc[:, predictands[1]].values
        best_params_15_45 = perform_grid_search(predictands[1], X_train, y_train)
        with open(params_path_15_45, 'w') as f:
            json.dump(best_params_15_45, f)

    return best_params_next_30, best_params_15_45, best_params_last_30


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog='xgb-training',
                                     description='XGB training script')
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -------------------------------------------------------------------------
    # Load data and derive persistence reference.
    # Predictors are all columns that are not target labels or split indicators.
    # The persistence forecast flags a storm as a hail producer if its current
    # MESH_max >= 35 mm (same threshold as the target labels).
    # -------------------------------------------------------------------------
    data = pd.read_csv(config['data_file'])
    #create 10 folds based on day of storm cells
    data = gen_kfolds_based_on_days(data, 10)
    # Calculate the persistance based reference, if MESH_max >= 35 then the persistence reference across
    # all three intervals is 1.
    data['persistence_bool'] = 0
    data.loc[data['MESH_max'] >= 35, 'persistence_bool'] = 1
    predictors =  ['MESH_area_ge_20', 'MESH_max', 'MESH_p90', 'storm_area', 'MUCAPE', 'MUCAPEm10m30',
                  'storm_duration', 'MUEL', 'ETH_p90', 'MUCIN', 'storm_area_trend', 'lightning_flash_rate',
                  'WBFZL', 'MESH_p90_trend', 'ETH_p90_trend', 'RH36mean', 'storm_motion_x',
                  'SRH03r', 'azshear_p90', 'U06mean', 'storm_motion_mag', 'SRH03l', 'MUVTEm20',
                  'lightning_flash_rate_trend', 'MESH_area_ge_20_trend', 'MESH_max_trend', 'deviant_motion_mag',
                  'storm_motion_y', 'V06mean', 'PW', 'deviant_motion_x', 'azshear_p90_trend',
                  'BWD06',
                  'deviant_motion_y', 'lightning_flash_rate_density_trend',
                  'lightning_flash_rate_density', 'LR36'
                  ]

    predictands = ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']

    save_path = config['save_path']
    if not os.path.exists(save_path):
        print(f"Directory {save_path} does not exist.")
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} exists.")

    results_df = []
    features_15 = []
    features_30 = []
    features_60 = []

    # Initialise output columns for storing out-of-fold predictions
    data['xgb_pred_15_45'] = ""
    data['xgb_pred_last30'] = ""
    data['xgb_pred_next30'] = ""

    # -------------------------------------------------------------------------
    # 10-fold cross-validation loop.
    # Three separate models are trained per fold, one per lead time.
    # Hyperparameters are loaded from JSON if available, otherwise computed
    # via GridSearchCV and saved for future runs.
    # -------------------------------------------------------------------------
    for n in range(0, 10):
        print("split = ", n)
        # Extract train/test splits for this fold and drop split indicator columns
        train_df = data.loc[data['split' + str(n)].str.contains('Train'), :]
        val_df = data.loc[data['split' + str(n)].str.contains('Test'), :]
        train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='split')))]
        val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='split')))]
        print("Training df size = ", len(list(train_df)), len(train_df))
        print("Validation df size = ", len(list(val_df)), len(val_df))

        # Load or compute hyperparameters for all three lead times
        next_30_params_path = os.path.join(save_path, 'best_params_next_30_split0.json')
        last_30_params_path = os.path.join(save_path, 'best_params_last_30_split0.json')
        params_path_15_45 = os.path.join(save_path, 'best_params_15_45_split0.json')

        best_params_next_30, best_params_15_45, best_params_last_30 = get_best_parameters(next_30_params_path,
                                                                                           params_path_15_45,
                                                                                           last_30_params_path,
                                                                                           train_df, predictors, predictands)

        # Shared feature and persistence arrays for this fold
        X_train = train_df.loc[:, predictors].values
        X_val = val_df.loc[:, predictors].values
        y_ref_TP = train_df.loc[:, 'persistence_bool']
        y_ref_VP = val_df.loc[:, 'persistence_bool']
        print("training/testing data shape:", X_train.shape, X_val.shape)

        # -----------------------------------------------------------------
        # 0-30 min model
        # -----------------------------------------------------------------
        print("\nTraining model: 0-30 minutes")
        y_train = train_df.loc[:, 'MESH_bool_next_30'].values
        y_val = val_df.loc[:, 'MESH_bool_next_30'].values
        results_first30, importances, y_pred = train_and_evaluate(X_train, y_train, X_val, y_val, y_ref_VP, predictors,
                                                                   best_params_next_30,
                                                                   'params/xgboost_first30_' + str(n) + '.json',
                                                                   'params/first30_split' + str(n), smote_flag=0)
        data.loc[val_df.index, "xgb_pred_next30"] = y_pred
        result_dict = {'split': n,
                       'MESH_next30_bsm': results_first30['bsm'],
                       'MESH_next30_bsref_c': results_first30['bsref_c'],
                       'MESH_next30_bss_c': results_first30['bss_c'],
                       'MESH_next30_bsref_p': results_first30['bsref_p'],
                       'MESH_next30_bss_p': results_first30['bss_p']}
        importances.to_csv("params/xgb_features_first30_" + str(n) + ".csv")
        print(importances)

        # -----------------------------------------------------------------
        # 15-45 min model
        # -----------------------------------------------------------------
        print("\nTraining model: 15-45 minutes")
        y_train = train_df.loc[:, 'MESH_bool_15_45'].values
        y_val = val_df.loc[:, 'MESH_bool_15_45'].values
        results_15_45, importances, y_pred = train_and_evaluate(X_train, y_train, X_val, y_val, y_ref_VP, predictors,
                                                                 best_params_15_45,
                                                                 'params/xgboost_15_45_' + str(n) + '.json',
                                                                 'params/first15_45_split' + str(n), smote_flag=0)
        data.loc[val_df.index, "xgb_pred_15_45"] = y_pred
        result_dict.update({'MESH_15_45_bsm': results_15_45['bsm'],
                            'MESH_15_45_bsref_c': results_15_45['bsref_c'],
                            'MESH_15_45_bss_c': results_15_45['bss_c'],
                            'MESH_15_45_bsref_p': results_15_45['bsref_p'],
                            'MESH_15_45_bss_p': results_15_45['bss_p']})
        importances.to_csv("params/xgb_features_15_45_" + str(n) + ".csv")

        # -----------------------------------------------------------------
        # 30-60 min model
        # -----------------------------------------------------------------
        print("\nTraining model: 30-60 minutes")
        y_train = train_df.loc[:, 'MESH_bool_last_30'].values
        y_val = val_df.loc[:, 'MESH_bool_last_30'].values
        results, importances, y_pred = train_and_evaluate(X_train, y_train, X_val, y_val, y_ref_VP, predictors,
                                                          best_params_last_30,
                                                          'params/xgboost_last30_' + str(n) + '.json',
                                                          'params/last_30_split' + str(n), smote_flag=0)
        data.loc[val_df.index, "xgb_pred_last30"] = y_pred
        result_dict.update({'MESH_last30_bsm': results['bsm'],
                            'MESH_last30_bsref_c': results['bsref_c'],
                            'MESH_last30_bss_c': results['bss_c'],
                            'MESH_last30_bsref_p': results['bsref_p'],
                            'MESH_last30_bss_p': results['bss_p']})
        importances.to_csv("params/xgb_features_last30_" + str(n) + ".csv")

        # Append results for this fold to the results CSV
        if n == 0:
            results_df = pd.DataFrame(result_dict, index=[n])
            results_df.to_csv('nowcasting_XGB_results.csv', index=False)
        else:
            results_df = pd.DataFrame(result_dict, index=[n])
            results_df.to_csv('nowcasting_XGB_results.csv', index=False, mode='a', header=False)

    # Save full out-of-fold predictions for all splits to CSV for figure generation
    data.to_csv('xgb_predictions.csv', index=False)