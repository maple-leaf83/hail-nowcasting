import os
import pandas as pd

import json
import shap
import logging
import tensorflow as tf
import argparse
import yaml
from utils import *
import shutil

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


def feature_selection(train_df, val_df, predictand, predictors, params, output_filename):
    y_train = train_df.loc[:, predictand,].values
    y_test = val_df.loc[:, predictand].values
    ## begin training with all features
    feature_num = 37
    X_train = train_df.loc[:, predictors].values
    X_test = val_df.loc[:, predictors].values
    while feature_num >= 1:
        ### Add specific values, rather than just passing in the best parameter list

        print("training/testing data shape:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        results_first30, xgb_model = train_for_feature_selection(X_train, y_train, X_test, y_test, predictors,
                                                                 params, smote_flag=0)
        # data.loc[val_df.index, "xgb_pred_next30"] = y_pred

        result_dict = {'split': n, 'num_features': feature_num,
                       predictand: results_first30['bss']}

        if n == 0 and feature_num == 37:
            results_df = pd.DataFrame(result_dict, index=[n])
            results_df.to_csv(output_filename, index=False)
        else:
            results_df = pd.DataFrame(result_dict, index=[n])
            results_df.to_csv(output_filename, index=False, mode='a', header=False)

        if feature_num == 1:
            break

        ## do shap analysis for last output: 30-60 minutes
        X_test = val_df.loc[:, predictors]
        explainer = shap.TreeExplainer(xgb_model, X_test)
        shap_values = explainer(X_test, check_additivity=False)
        # print(shap_values)
        shap_features = predictors
        print("shap_features = ", shap_features)
        shap_abs = np.mean(shap_values.abs.values, axis=0)
        shap_sorted = sorted(shap_abs, reverse=True)
        index = np.argsort(shap_abs)[::-1]

        features_sorted = [shap_features[i] for i in index]
        # droping last 5% of features
        shap_cumsum = np.cumsum(shap_sorted)
        feature_index = [i for i, s in enumerate(shap_cumsum) if s <= 0.95 * np.max(shap_cumsum)]
        if len(features_sorted) == 2:
            feature_num = 1
        else:
            feature_num = feature_index[-1]
        print("New # features = ", feature_num)
        predictors_updated = features_sorted[0:feature_num]
        print("predictors_updated = ", predictors_updated)
        predictors = predictors_updated
        # update training and testing X matrices
        X_train = train_df.loc[:, predictors].values
        X_test = val_df.loc[:, predictors].values



if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='xgb-training',
                                     description='XGB training script')
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = pd.read_csv(config['data_file'])

    # List of predictands
    predictands = ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']
    save_path = 'params/'

    # Check if the directory exists
    if not os.path.exists(save_path):
        print(f"Directory {save_path} does not exist.")
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} exists.")
    results_df = []


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
    n = 0 # select one split, here we used 0
    print("split = ", n)
    train_df = data.loc[data['split'+str(n)].str.contains('Train'), :]
    val_df = data.loc[data['split' + str(n)].str.contains('Test'), :]
    train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='split')))]
    val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='split')))]
    print("Training df size = ", len(list(train_df)), len(train_df))
    print("Validation df size = ", len(list(val_df)), len(val_df))

    # Check if the JSON files exist
    print("NEXT 30")
    next_30_params_path = os.path.join(save_path, 'best_params_next_30_split0.json')
    with open(next_30_params_path, 'r') as f:
        best_params_next_30 = json.load(f)
    feature_selection(train_df, val_df, 'MESH_bool_next_30', predictors, best_params_next_30,
                      'xgb_features_next30.csv')
    print("15-45")
    params_path_15_45 = os.path.join(save_path, 'best_params_15_45_split0.json')
    with open(params_path_15_45, 'r') as f:
        best_params_15_45 = json.load(f)
    feature_selection(train_df, val_df, 'MESH_bool_15_45', predictors, best_params_15_45,
                      'xgb_features_15.csv')
    print("LAST 30")
    last_30_params_path = os.path.join(save_path, 'best_params_last_30_split0.json')
    with open(last_30_params_path, 'r') as f:
        best_params_last_30 = json.load(f)
    feature_selection(train_df, val_df, 'MESH_bool_last_30', predictors, best_params_last_30,
                      'xgb_features_last30.csv')

