"""
utils.py
--------
Shared utility functions for training and evaluating XGBoost and MLP models
for severe hail nowcasting. Includes Brier score computation, XGBoost training
and evaluation, hyperparameter search, and feature importance extraction.
"""

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, make_scorer, brier_score_loss, \
    roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
# import cupy as cp
import pickle
import pandas as pd


def get_feature_importance(bst):
    """
    Extract feature importance scores from a trained XGBoost booster.

    Computes five importance metrics:
        - weight:      Number of times each feature is used to split across all trees.
        - gain:        Average information gain per split for each feature.
        - cover:       Average number of samples affected by splits on each feature.
        - total_gain:  Total information gain across all splits for each feature.
        - total_cover: Total number of samples affected across all splits.

    Args:
        bst (xgb.Booster): Trained XGBoost booster (not XGBClassifier).

    Returns:
        dict: Keys are importance type strings ('weight', 'gain', 'cover',
              'totalgain', 'totalcover'); values are dicts mapping feature
              names to scores.
    """
    # Get feature importance scores
    importance_weight = bst.get_score(importance_type = 'weight')
    importance_gain = bst.get_score(importance_type='gain')
    importance_cover = bst.get_score(importance_type='cover')
    importance_totalgain = bst.get_score(importance_type='total_gain')
    importance_totalcover = bst.get_score(importance_type='total_cover')
    feature_imps = {'weight': importance_weight,
                    'gain': importance_gain,
                    'cover': importance_cover,
                    'totalgain': importance_totalgain,
                    'totalcover': importance_totalcover,}
    return feature_imps


def compute_brier_scores(y_val, y_pred):
    """
    Compute the Brier Score and Brier Skill Score (BSS) relative to climatology.

    The reference forecast is sample climatology: a constant forecast equal to
    the observed event frequency. BSS > 0 indicates skill over climatology.

    Note: Use compute_BSS_scores for BSS relative to a persistence forecast.

    Args:
        y_val (np.array): True binary labels (0 or 1).
        y_pred (np.array): Predicted probabilities in [0, 1].

    Returns:
        bs_model (float): Brier Score of the model. Lower is better.
        bs_ref (float):   Brier Score of the climatology reference.
        bss (float):      Brier Skill Score = 1 - (bs_model / bs_ref).
                          Ranges from -inf to 1; 1 is perfect, 0 is no skill.
    """
    # Calculate the Brier Score for the model's predictions
    bs_model = brier_score_loss(y_val, y_pred)

    # Calculate the reference forecast (sample climatology)
    # Sample climatology is the mean of the true labels
    ybar = np.mean(y_val)

    # Create an array with the same shape as y_val, filled with the mean value
    y_ref = np.full_like(y_val, ybar)

    # Calculate the Brier Score for the reference forecast
    bs_ref = brier_score_loss(y_val, y_ref)

    # Calculate the Brier Skill Score (BSS)
    # BSS = 1 - (Brier Score of the model / Brier Score of the reference forecast)
    bss = 1 - (bs_model / bs_ref)

    return bs_model, bs_ref, bss


def compute_BSS_scores(y_true, y_pred, y_ref):
    """
    Compute the Brier Score and Brier Skill Scores relative to both climatology
    and a persistence forecast.

    The climatology reference uses the observed event frequency (sample mean)
    as a constant forecast. The persistence reference (y_ref) uses current storm
    state — a storm is considered a hail producer if its current MESH >= 35 mm.

    Args:
        y_true (np.array): True binary labels (0 or 1).
        y_pred (np.array): Predicted probabilities in [0, 1].
        y_ref  (np.array): Persistence forecast probabilities (binary 0/1),
                           derived from current MESH threshold.

    Returns:
        bs_model (float):          Brier Score of the model. Lower is better.
        bs_ref_climatology (float): Brier Score of the climatology reference.
        bs_ref_persistence (float): Brier Score of the persistence reference.
        bss_c (float): BSS relative to climatology = 1 - (bs_model / bs_ref_climatology).
        bss_p (float): BSS relative to persistence = 1 - (bs_model / bs_ref_persistence).
                       Both skill scores: 1 is perfect, 0 is no skill over the reference.
    """
    # Calculate the Brier Score for the model's predictions
    bs_model = brier_score_loss(y_true, y_pred)


    # calculaye brier score using mean of true values, i.e., climatology reference model
    bs_ref_climatology = brier_score_loss(y_true, np.full_like(y_true, np.mean(y_true)))

    # Calculate the Brier Score for the reference forecast
    bs_ref_persistence = brier_score_loss(y_true, y_ref)

    # Calculate the Brier Skill Score (BSS)
    # BSS = 1 - (Brier Score of the model / Brier Score of the reference forecast)
    bss_c = 1 - (bs_model / bs_ref_climatology)
    bss_p = 1- (bs_model / bs_ref_persistence)

    return bs_model, bs_ref_climatology, bs_ref_persistence, bss_c, bss_p

def bss_scorer(y_true, y_pred):
    """
    Sklearn-compatible scorer wrapping compute_brier_scores.

    Used as a custom scoring function in GridSearchCV during XGBoost
    hyperparameter search. Returns BSS relative to climatology.

    Args:
        y_true (np.array): True binary labels.
        y_pred (np.array): Predicted probabilities.

    Returns:
        bss (float): Brier Skill Score relative to climatology.
    """
    _, _, bss = compute_brier_scores(y_true, y_pred)
    return bss


def weighted_binary_cross_entropy(preds, dtrain, weight):
    """
    Calculate the gradient and hessian for a weighted binary cross-entropy loss function.

    Args:
    preds (np.array): Predicted values (logits).
    dtrain (xgb.DMatrix): Training data.
    weight (float): Weight for the positive class.

    Returns:
    grad (np.array): Gradient of the loss function.
    hess (np.array): Hessian of the loss function.
    """
    # Get the true labels from the training data
    labels = dtrain.get_label()

    # Apply the sigmoid function to transform logits to probabilities
    # Sigmoid function: σ(z) = 1 / (1 + exp(-z))
    preds = 1.0 / (1.0 + np.exp(-preds))

    # Define weights using the optimal weight parameter
    # Assign weight to positive class (label = 1) and 1.0 to negative class (label = 0)
    weights = np.where(labels == 1, weight, 1.0)

    # Calculate the gradient of the weighted binary cross-entropy loss
    # Gradient of cross-entropy loss: ∂L/∂z = σ(z) - y
    # Weighted gradient: grad = (σ(z) - y) * weight
    grad = (preds - labels) * weights

    # Calculate the hessian (second derivative) of the weighted binary cross-entropy loss
    # Hessian of cross-entropy loss: ∂²L/∂z² = σ(z) * (1 - σ(z))
    # Weighted hessian: hess = σ(z) * (1 - σ(z)) * weight
    hess = preds * (1.0 - preds) * weights

    return grad, hess


def perform_grid_search(predictands, X_train, y_train):
    """
    Perform GridSearchCV to find optimal XGBoost hyperparameters for a single target.

    Searches over max_depth, subsample, alpha (L1), and lambda (L2) using
    3-fold cross-validation scored by BSS. Uses a random 80/20 train/val split
    before searching. Runs on GPU (device='cuda') if available.

    Args:
        predictands (str):      Name of the target variable (used for labelling only).
        X_train (np.ndarray):   Feature matrix, shape (N, n_features).
        y_train (np.ndarray):   Binary target labels, shape (N,).

    Returns:
        best_params (dict): Best hyperparameter values found by GridSearchCV.
    """
    x_val_index = np.random.choice(range(X_train.shape[0]), size=(int)(0.2 * X_train.shape[0]), replace=False)
    x_train_index = [i for i in range(X_train.shape[0]) if i not in x_val_index]

    X_train = X_train[x_train_index, :]
    y_train = y_train[x_train_index]
    best_params_dict = {}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initial parameters for the model
    params = {
        'objective': 'logloss',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,  # This will be overridden by grid search
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # Define parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [30, 40, 50, 60],
        'subsample': [0.3, 0.6, 0.8, 1.0],
        # 'scale_pos_weight': [9, 11],
        'alpha': [0.5, 1.0, 2.0],
        'lambda': [ 0.5, 1.0, 2.0]
    }

    # Custom scorer to use Brier Skill Score (BSS)
    scorer = make_scorer(bss_scorer, greater_is_better=True)

    # Perform grid search
    X_train = np.array(X_train)
    # y_train = cp.array(y_train)
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(**params, device='cuda'), param_grid=param_grid,
                               scoring=scorer, cv=3, verbose=4, n_jobs=-2)

    # Fit the model using grid search
    grid_search.fit(X_train, y_train)

    # Retrieve the best parameters from the grid search
    best_params = grid_search.best_params_

    return best_params


def train_and_evaluate(X_train, y_train, X_test, y_test, y_test_refP, predictors,
                       best_params_dict, save_model_fname, loss_filename, smote_flag):
    """
    Train a single XGBoost model and evaluate it on a held-out test set.

    Splits X_train into an internal 80/20 train/validation set for early stopping.
    Optionally applies SMOTE oversampling to address class imbalance. Features are
    standardised using StandardScaler fitted on training data only.

    The positive class weight (scale_pos_weight=11) is set to address class
    imbalance in the training data.

    Args:
        X_train (np.ndarray):       Training features, shape (N_train, n_features).
        y_train (np.ndarray):       Training labels, shape (N_train,).
        X_test (np.ndarray):        Test features, shape (N_test, n_features).
        y_test (np.ndarray):        Test labels, shape (N_test,).
        y_test_refP (np.ndarray):   Persistence forecast for the test set (binary),
                                    used as BSS reference.
        predictors (list):          Feature names, used for labelling importance output.
        best_params_dict (dict):    XGBoost hyperparameters (from perform_grid_search
                                    or pre-saved JSON).
        save_model_fname (str):     File path to save the trained model (.json).
        loss_filename (str):        Label string for logging (printed only).
        smote_flag (int):           If 1, apply SMOTE oversampling before training.

    Returns:
        results (dict):     Brier scores and skill scores:
                            {'bsm', 'bsref_c', 'bss_c', 'bsref_p', 'bss_p'}.
        scores_df (pd.DataFrame): Feature importances (weight, gain, cover,
                                  total_gain, total_cover) for each predictor.
        y_pred_proba (np.ndarray): Predicted probabilities on the test set.
    """
    print(loss_filename)
    x_val_index = np.random.choice(range(X_train.shape[0]), size= (int)(0.2*X_train.shape[0]), replace=False)
    x_train_index = [i for i in range(X_train.shape[0]) if i not in x_val_index]

    X_val = X_train[x_val_index, :]
    y_val = y_train[x_val_index]

    X_train = X_train[x_train_index,:]
    y_train = y_train[x_train_index]
    print(X_train.shape, X_val.shape, X_test.shape)
    if smote_flag:
        # Standardize the features after resampling
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    best_params = best_params_dict
    # best_params['eval_metric'] = 'logloss'
    best_params['scale_pos_weight'] = 11.0
    optimal_weight = best_params['scale_pos_weight']
    best_params['objective'] = 'binary:logistic'

    evals_result = {}

    # Train final model on the entire training data
    bst_final = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    bst_final = xgb.train(
        best_params,
        xgb.DMatrix(X_train_res, label=y_train),
        num_boost_round=1000,
        evals=[(xgb.DMatrix(X_train_res, label=y_train), 'train'),
               (xgb.DMatrix(X_val, label=y_val), 'validation')],
        early_stopping_rounds=200,
        evals_result=evals_result,
        verbose_eval=False,
        # obj= lambda preds, dtrain: weighted_binary_cross_entropy(preds, dtrain, optimal_weight)
    )
    print("Training complete")
    y_pred_proba = bst_final.predict(xgb.DMatrix(X_test), output_margin=False)
    print(np.min(y_pred_proba), np.max(y_pred_proba))

    # save model
    bst_final.save_model(save_model_fname)

    """ --- get feature importance """
    importances = get_feature_importance(bst_final)
    print(importances.keys())
    """ create df with importances"""
    scores_df = pd.DataFrame()
    scores_df['importance'] = ""
    for p in predictors:
        scores_df[p] = ""

    for key in importances.keys():
        len_df = len(scores_df)
        scores_df.loc[len(scores_df), "importance"] = key
        for i in range(len(predictors)):
            scores_df.loc[len_df, predictors[i]] = importances[key].get('f'+str(i))
        len_df += 1

    bs_model, bs_ref_climatology, bs_ref_persistence, bss_c, bss_p = compute_BSS_scores(y_test, y_pred_proba, y_test_refP)
    # print(f"Brier Score (Model): {bs_model}")
    # print(f"Brier Score (Reference): {bs_ref}")
    print(f"Brier Skill Score (BSS-P): {bss_p}")
    print(f"Brier Skill Score (BSS-C): {bss_c}")
    results = {'bsm': bs_model, 'bsref_c': bs_ref_climatology, 'bss_c': bss_c, 'bsref_p': bs_ref_persistence, 'bss_p': bss_p}

    return results, scores_df, y_pred_proba

def train_for_feature_selection(X_train, y_train, X_test, y_test, predictors,
                       best_params_dict, smote_flag):
    """
    Train an XGBoost model for use within recursive feature selection.

    Equivalent to train_and_evaluate but without saving the model to disk or
    computing per-output SHAP importances. Returns BSS relative to climatology
    only (no persistence reference required), keeping the interface lightweight
    for iterative feature selection loops.

    Args:
        X_train (np.ndarray):     Training features, shape (N_train, n_features).
        y_train (np.ndarray):     Training labels, shape (N_train,).
        X_test (np.ndarray):      Test features, shape (N_test, n_features).
        y_test (np.ndarray):      Test labels, shape (N_test,).
        predictors (list):        Feature names (used for logging only).
        best_params_dict (dict):  XGBoost hyperparameters.
        smote_flag (int):         If 1, apply SMOTE oversampling before training.

    Returns:
        results (dict):           {'bsm', 'bsref', 'bss'} — Brier score and BSS
                                  relative to climatology.
        bst_final (xgb.Booster): Trained XGBoost booster for SHAP analysis.
    """

    x_val_index = np.random.choice(range(X_train.shape[0]), size= (int)(0.2*X_train.shape[0]), replace=False)
    x_train_index = [i for i in range(X_train.shape[0]) if i not in x_val_index]

    X_val = X_train[x_val_index, :]
    y_val = y_train[x_val_index]

    X_train = X_train[x_train_index,:]
    y_train = y_train[x_train_index]
    print(X_train.shape, X_val.shape, X_test.shape)
    if smote_flag:
        # Standardize the features after resampling
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    best_params = best_params_dict
    # best_params['eval_metric'] = 'logloss'
    best_params['scale_pos_weight'] = 11.0
    optimal_weight = best_params['scale_pos_weight']
    best_params['objective'] = 'binary:logistic'
    evals_result = {}

    # Train final model on the entire training data
    bst_final = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    bst_final = xgb.train(
        best_params,
        xgb.DMatrix(X_train_res, label=y_train),
        num_boost_round=1000,
        evals=[(xgb.DMatrix(X_train_res, label=y_train), 'train'),
               (xgb.DMatrix(X_val, label=y_val), 'validation')],
        early_stopping_rounds=200,
        evals_result=evals_result,
        verbose_eval=False,
        # obj= lambda preds, dtrain: weighted_binary_cross_entropy(preds, dtrain, optimal_weight)
        # objective='binary:logistic',
    )


    print("Training complete")
    y_pred_proba = bst_final.predict(xgb.DMatrix(X_test), output_margin=False)
    bs_model, bs_ref, bss = compute_brier_scores(y_test, y_pred_proba)
    print(f"Brier Score (Model): {bs_model}")
    print(f"Brier Score (Reference): {bs_ref}")
    print(f"Brier Skill Score (BSS): {bss}")
    results = {'bsm': bs_model, 'bsref': bs_ref, 'bss': bss}

    return results, bst_final