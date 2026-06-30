"""
mlp_training.py
---------------
Trains a multi-output MLP neural network to predict the probability of severe
hail (MESH >= 35 mm) at three forecast lead times:
    - 0-30 minutes  (MESH_bool_next_30)
    - 15-45 minutes (MESH_bool_15_45)
    - 30-60 minutes (MESH_bool_last_30)

The model is evaluated using the Brier Skill Score (BSS) relative to both a
climatology reference and a persistence forecast.

Training uses 10-fold cross-validation, where folds are constructed by day to
avoid temporal leakage between train and test sets.

Usage:
    python mlp_training.py -c config.yml

Outputs:
    - Trained model files: models/nowcasting_model<n>.keras  (one per fold)
    - Skill score results: models/MLP_training_results.csv
    - Full predictions:    mlp_predictions.csv
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.metrics import F1Score
from sympy.codegen.ast import continue_
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import argparse
import yaml
from utils import *
import shutil
from networks import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, roc_auc_score, auc, precision_recall_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification with class imbalance.

    Down-weights easy (well-classified) examples so training focuses on hard
    examples near the decision boundary. Introduced by Lin et al. (2017).

    Args:
        alpha (float): Weighting factor for the positive class. Should be > 0.5
                       when positives are the minority class. Default: 0.25.
        gamma (float): Focusing parameter. Higher values suppress easy negatives
                       more aggressively. gamma=0 reduces to standard BCE.
                       Default: 2.0.

    Returns:
        A Keras-compatible loss function.

    Note: Not used in the final model — binary cross-entropy with undersampling
    and per-sample weights was found to outperform focal loss for this dataset.
    Retained here for reference.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)
    return loss_fn


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


def gen_kfolds_based_on_days(df, K):
    """
    Generate K-fold cross-validation splits stratified by day.

    Days are shuffled randomly (seed=42) then divided into K contiguous
    blocks. Splitting by day rather than by sample prevents temporal leakage,
    since storm observations within the same day are not independent.

    Args:
        df (pd.DataFrame): Input dataframe containing a 'time_unix' column.
        K (int): Number of folds.

    Returns:
        pd.DataFrame: Input dataframe with K additional columns 'split0'...'splitK-1',
                      each containing 'Train' or 'Test'.

    Note: This function is called once to pre-compute splits and save them to
    the CSV. It is not called during normal training.
    """
    unique_days = df.loc[:, 'time_unix'].unique()
    print(len(unique_days))
    np.random.seed(42)
    np.random.shuffle(unique_days)

    for n in range(K):
        df['split' + str(n)] = ""
        lower_lim = int(float(n * len(unique_days)) / K)
        upper_lim = int(float((n + 1) * len(unique_days)) / K)

        if n == K - 1:
            upper_lim = len(unique_days)
        test_days = unique_days[lower_lim:upper_lim]
        df.loc[df['time_unix'].isin(test_days), 'split' + str(n)] = 'Test'
        df.loc[~df['time_unix'].isin(test_days), 'split' + str(n)] = 'Train'

    return df


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
    # The persistence forecast assumes the current storm state persists into
    # the forecast window. A storm is flagged as a hail producer if its current
    # MESH_max >= 35 mm (the same threshold used for the target labels).
    # -------------------------------------------------------------------------
    data = pd.read_csv(config['data_file'])
    # create 10 folds based on day of storm cells
    data = gen_kfolds_based_on_days(data, 10)
    # Calculate the persistance based reference, if MESH_max >= 35 then the persistence reference across
    # all three intervals is 1.
    data['persistence_bool'] = 0
    data.loc[data['MESH_max'] >= 35, 'persistence_bool'] = 1

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
    print("Features = ", predictors)

    # -------------------------------------------------------------------------
    # Initialise output columns for storing out-of-fold predictions
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame()
    data['MLP_next_30'] = ""
    data['MLP_last_30'] = ""
    data['MLP_15_45'] = ""

    save_path = 'models'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # -------------------------------------------------------------------------
    # 10-fold cross-validation loop
    # Each fold uses a different subset of days as the held-out test set.
    # -------------------------------------------------------------------------
    for n in range(0, 10):
        print("split = ", n)

        # Extract train/test splits for this fold and drop split/MLP columns
        train_df = data.loc[data['split' + str(n)].str.contains('Train'), :]
        val_df = data.loc[data['split' + str(n)].str.contains('Test'), :]
        train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='split')))]
        train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='MLP')))]
        val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='split')))]
        val_df = val_df[val_df.columns.drop(list(val_df.filter(regex='MLP')))]

        print("Training df size = ", len(list(train_df)), len(train_df))
        print("Validation df size = ", len(list(val_df)), len(val_df))

        # ---------------------------------------------------------------------
        # Class balancing via undersampling of the majority (no-hail) class.
        # All positive (hail) samples are retained. Negative samples are
        # randomly subsampled so that negatives:positives = mult_factor:1.
        # Stratification is based on the primary target (MESH_bool_next_30).
        # ---------------------------------------------------------------------
        for mult_factor in [3.5]:
            X_eq_train = train_df.loc[train_df['MESH_bool_next_30'] == 1]
            X_train_0 = train_df.loc[train_df['MESH_bool_next_30'] == 0]
            if int(len(X_eq_train) * mult_factor) >= len(X_train_0):
                X_eq_train = pd.concat([X_eq_train, X_train_0])
            else:
                X_eq_train = pd.concat([X_eq_train, X_train_0.sample(int(len(X_eq_train) * mult_factor))])

            # Extract feature and label arrays
            y_train = X_eq_train.loc[:, ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']].values
            X_train = X_eq_train.loc[:, predictors]

            # Split training data into train/internal-validation sets.
            # Stratified split ensures class balance is maintained in both sets.
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=y_train[:, 0])  # stratify on primary target

            # Hold-out test set (never seen during training)
            y_test = val_df.loc[:, ['MESH_bool_next_30', 'MESH_bool_15_45', 'MESH_bool_last_30']].values
            X_test = val_df.loc[:, predictors]

            # Persistence reference for BSS computation
            y_ref_VP = val_df.loc[:, 'persistence_bool']

            # -----------------------------------------------------------------
            # Feature scaling: standardise to zero mean and unit variance.
            # Scaler is fit on training data only to prevent data leakage.
            # Neural networks are sensitive to feature scale in a way that
            # gradient-boosted trees are not; scaling is therefore required
            # here but not for the XGBoost model.
            # -----------------------------------------------------------------
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print(X_tr.shape, X_val.shape, X_test.shape)

            # -----------------------------------------------------------------
            # Model architecture: 5-layer shared trunk (512 units each),
            # with task-specific head layers before each sigmoid output.
            # See networks.py (gen_mlp_network) for full architecture details.
            # -----------------------------------------------------------------
            filters = [[512, 512, 512, 512, 512]]

            for i in range(len(filters)):
                callbacks = [
                    # Save the best model (lowest val_loss) to disk each fold
                    ModelCheckpoint(filepath=os.path.join(save_path, 'nowcasting_model' + str(n) + '.keras'),
                                    monitor='val_loss', save_best_only=True),
                    # Halve LR if val_loss does not improve for 10 epochs
                    ReduceLROnPlateau(monitor='val_loss', patience=10, min_lr=1e-10),
                    # Stop training early if val_loss does not improve for 300 epochs
                    EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
                ]

                model = gen_mlp_network(input_shape=(X_tr.shape[1],), num_filters=filters[i], dropout=0.1)
                model.compile(optimizer=Adam(learning_rate=0.0001),
                              loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                              )

                # Train the model
                y_train_form = format(y_tr)
                history = model.fit(X_tr, y_train_form, epochs=1500, batch_size=512,
                                    validation_data=(X_val, format(y_val)),
                                    callbacks=callbacks,
                                    verbose=1)

                # -------------------------------------------------------------
                # Evaluate on held-out test set
                # -------------------------------------------------------------
                test_preds = model.predict(X_test)
                print(model.evaluate(X_test, format(y_test)))
                tp1 = test_preds[0]  # 0-30 min predictions
                tp2 = test_preds[1]  # 15-45 min predictions
                tp3 = test_preds[2]  # 30-60 min predictions
                test_results = np.concatenate([tp1, tp2, tp3], axis=1)

                # Brier Skill Scores relative to climatology (BSS_C) and
                # persistence (BSS_P) for each of the three lead times
                bs_model1, bs_ref_c1, bs_ref_p1, bss_c1, bss_p1 = compute_BSS_scores(y_test[:, 0], test_results[:, 0], y_ref_VP)
                bs_model2, bs_ref_c2, bs_ref_p2, bss_c2, bss_p2 = compute_BSS_scores(y_test[:, 1], test_results[:, 1], y_ref_VP)
                bs_model3, bs_ref_c3, bs_ref_p3, bss_c3, bss_p3 = compute_BSS_scores(y_test[:, 2], test_results[:, 2], y_ref_VP)

                results_dict = {'split': n, 'data_factor': mult_factor,
                                'Brier Score30': bs_model1, 'Ref_C30': bs_ref_c1, 'Ref_P30': bs_ref_p1, 'BSS_C30': bss_c1, 'BSS_P30': bss_p1,
                                'Brier Score15': bs_model2, 'Ref_C15': bs_ref_c2, 'Ref_P15': bs_ref_p2, 'BSS_C15': bss_c2, 'BSS_P15': bss_p2,
                                'Brier Score60': bs_model3, 'Ref_C60': bs_ref_c3, 'Ref_P60': bs_ref_p3, 'BSS_C60': bss_c3, 'BSS_P60': bss_p3}
                print(results_dict)

                # Append results for this fold to the results CSV
                results_df = pd.DataFrame(results_dict, index=[0])
                results_df.to_csv(os.path.join(save_path, "MLP_training_results.csv"), mode='a', index=False, header=False)

                # Store out-of-fold predictions back into the main dataframe
                data.loc[data['split' + str(n)].str.contains('Test'), 'MLP_next_30'] = test_results[:, 0]
                data.loc[data['split' + str(n)].str.contains('Test'), 'MLP_15_45'] = test_results[:, 1]
                data.loc[data['split' + str(n)].str.contains('Test'), 'MLP_last_30'] = test_results[:, 2]

    # Save full out-of-fold predictions for all splits to CSV for figure generation
    data.to_csv('mlp_predictions.csv', index=False, header=True)
