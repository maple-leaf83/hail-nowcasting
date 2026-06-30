"""
gen_predictive_curves_mlp.py
-----------------------------
Generates verification figures comparing MLP and XGBoost hail nowcasting models:
    - Reliability diagrams (calibration curves) with sharpness histograms
      for all three forecast lead times (0-30, 15-45, 30-60 min)
    - ROC curves with AUC scores for all three lead times

Input CSVs must contain out-of-fold predictions from mlp_training.py and
xgboost_training.py respectively. See README for expected column names.

Usage:
    python gen_predictive_curves_mlp.py --mlp mlp_predictions.csv --xgb xgb_predictions.csv

Outputs:
    - results_figs/reldiagrams_hist<date>.pdf  Reliability diagrams + sharpness histograms
    - results_figs/aucplots_<date>.pdf         ROC curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, roc_curve
import seaborn as sns
from labellines import labelLines
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import ultraplot as uplt
import matplotlib.patches as mpatches



def gen_auc_plots(mlp_label, mlp_preds, xgb_label, xgb_preds, save_filename):
    """
    Plot ROC curves for MLP and XGBoost on a single axis and save to file.

    Points are subsampled from the full ROC curve for readability (every 1500th
    point for MLP, every 3000th for XGBoost). AUC scores are shown in the legend.

    Args:
        mlp_label (array-like):  True binary labels for MLP evaluation.
        mlp_preds (array-like):  MLP predicted probabilities.
        xgb_label (array-like):  True binary labels for XGBoost evaluation.
        xgb_preds (array-like):  XGBoost predicted probabilities.
        save_filename (str):     Output file path for the saved figure.

    Returns:
        None. Saves figure to save_filename.
    """
    cp = sns.color_palette("colorblind", n_colors=11)
    fig, ax = plt.subplots(figsize=(5,5))
    fpr_m, tpr_m, thresh = roc_curve(mlp_label, mlp_preds)
    fpr_m_points = fpr_m[0::1500]
    fpr_m_points = np.append(fpr_m_points, fpr_m[-1])
    tpr_m_points = tpr_m[0::1500]
    tpr_m_points = np.append(tpr_m_points, tpr_m[-1])
    mlp_auc = roc_auc_score(mlp_label, mlp_preds)
    print("MLP AUC = ", roc_auc_score(mlp_label, mlp_preds))

    # XGBoost
    fpr_x, tpr_x, thresh_x = roc_curve(xgb_label, xgb_preds)
    fpr_x_points = fpr_x[0::3000]
    fpr_x_points = np.append(fpr_x_points, fpr_x[-1])
    tpr_x_points = tpr_x[0::3000]
    tpr_x_points = np.append(tpr_x_points, tpr_x[-1])
    print("XGB AUC = ", roc_auc_score(xgb_label, xgb_preds))
    xgb_auc = roc_auc_score(xgb_label, xgb_preds)
    print(fpr_x.shape, tpr_x.shape)

    l1, = plt.plot(fpr_m_points, tpr_m_points, marker='o', markersize =5, linestyle='-', color=cp[1], label='MLP')
    l2, = plt.plot(fpr_x_points, tpr_x_points, marker='o', markersize =5, linestyle='-', color=cp[2], label='XGBoost')
    plt.plot([0, 1], [0, 1.0], color='k', linestyle='-')

    vals = np.arange(0.45, 0, -0.05)
    # labelLines(lines, align=True, xvals=vals)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.grid(True, linestyle='--', color='0.75')
    plt.legend(handles=[l1, l2], labels=[f'MLP, AUC = {mlp_auc:.3f}', f'XGBoost, AUC = {xgb_auc:.3f}'], fontsize=12)
    plt.savefig(save_filename, bbox_inches='tight')

def gen_auc_metrics(mlp_label, mlp_preds, xgb_label, xgb_preds):
    """
    Compute ROC curve points and AUC scores for MLP and XGBoost.

    Returns subsampled ROC curve arrays suitable for plotting, rather than
    saving a figure directly. Used by the multi-panel AUC figure in __main__.

    Args:
        mlp_label (array-like): True binary labels for MLP evaluation.
        mlp_preds (array-like): MLP predicted probabilities.
        xgb_label (array-like): True binary labels for XGBoost evaluation.
        xgb_preds (array-like): XGBoost predicted probabilities.

    Returns:
        mlp_auc (float):          AUC score for the MLP.
        fpr_m_points (np.ndarray): Subsampled false positive rates for MLP.
        tpr_m_points (np.ndarray): Subsampled true positive rates for MLP.
        xgb_auc (float):          AUC score for XGBoost.
        fpr_x_points (np.ndarray): Subsampled false positive rates for XGBoost.
        tpr_x_points (np.ndarray): Subsampled true positive rates for XGBoost.
    """
    cp = sns.color_palette("colorblind", n_colors=11)
    fig, ax = plt.subplots(figsize=(5,5))
    fpr_m, tpr_m, thresh = roc_curve(mlp_label, mlp_preds)
    fpr_m_points = fpr_m[0::1500]
    fpr_m_points = np.append(fpr_m_points, fpr_m[-1])
    tpr_m_points = tpr_m[0::1500]
    tpr_m_points = np.append(tpr_m_points, tpr_m[-1])
    mlp_auc = roc_auc_score(mlp_label, mlp_preds)
    print("MLP AUC = ", roc_auc_score(mlp_label, mlp_preds))

    # XGBoost
    fpr_x, tpr_x, thresh_x = roc_curve(xgb_label, xgb_preds)
    fpr_x_points = fpr_x[0::3000]
    fpr_x_points = np.append(fpr_x_points, fpr_x[-1])
    tpr_x_points = tpr_x[0::3000]
    tpr_x_points = np.append(tpr_x_points, tpr_x[-1])
    print("XGB AUC = ", roc_auc_score(xgb_label, xgb_preds))
    xgb_auc = roc_auc_score(xgb_label, xgb_preds)
    print(fpr_x.shape, tpr_x.shape)

    return mlp_auc, fpr_m_points, tpr_m_points, xgb_auc, fpr_x_points, tpr_x_points

def plot_reliability_diagram(y_true_mlp, y_prob_mlp, y_true_xgb, y_prob_xgb, n_bins=10, save_fname="temp.pdf"):
    """
    Plot a reliability diagram (calibration curve) with a sharpness histogram
    for both MLP and XGBoost, and save to file.

    The reliability diagram shows observed event frequency vs. predicted
    probability in uniform-width bins. The diagonal represents perfect
    calibration. The histogram below shows the distribution of predicted
    probabilities (sharpness) on a log scale.

    XGBoost predictions are clipped to [1e-5, 0.99945] to avoid log(0) errors
    in the calibration curve computation.

    Args:
        y_true_mlp (array-like): True binary labels for MLP evaluation.
        y_prob_mlp (array-like): MLP predicted probabilities.
        y_true_xgb (array-like): True binary labels for XGBoost evaluation.
        y_prob_xgb (array-like): XGBoost predicted probabilities.
        n_bins (int):            Number of uniform-width bins for calibration curve. Default: 10.
        save_fname (str):        Output file path. The histogram is saved with
                                 'hist' appended before the extension. Default: 'temp.pdf'.

    Returns:
        None. Saves figure to save_fname (with 'hist' suffix).
    """
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 8))
    cp = sns.color_palette("colorblind", n_colors=11)
    # Use scikit-learn's calibration_curve to get the fractions of positives and mean probabilities
    prob_true_mlp, prob_pred_mlp = calibration_curve(y_true_mlp, y_prob_mlp, n_bins=n_bins, strategy='uniform')
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_true_xgb, np.clip(y_prob_xgb, a_min=0.00001, a_max=0.99945), n_bins=n_bins, strategy='uniform')

    noskill_x = []
    noskill_y = []

    for x in np.arange(0, 1.10, 0.1):
        noskill_x.append(x)
        noskill_y.append(0.5*(np.mean(y_true_mlp) + x))

    # Plot the reliability diagram
    ax.plot(prob_pred_mlp, prob_true_mlp, marker='s', label='MLP', color=cp[1])
    ax.plot(prob_pred_xgb, prob_true_xgb, marker='s', label='XGBoost', color=cp[2])
    ax.plot([0, 1], [0, 1], color='k')
    ax.set_xlabel('Mean Predicted Probability', fontsize=18)
    ax.set_ylabel('Fraction of Positives', fontsize=18)
    # ax.set_title('Reliability Diagram (Calibration Plot)', fontsize=14)

    # Add a legend
    ax.legend(loc='upper left', fontsize=14)

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', color='0.75')
    ax.set_aspect('equal', adjustable='box')

    ax_hist = ax.inset_axes([0, -0.2, 1, 0.2], transform=ax.transAxes)

    ax_hist.hist(y_prob_mlp, bins=n_bins, edgecolor='black', alpha=0.5, color=cp[1], log=True)
    ax_hist.hist(np.clip(y_prob_xgb, a_min=0.00001, a_max=0.99945), bins=n_bins,
                 edgecolor='black', alpha=0.5, color=cp[2], log=True)
    ax_hist.set_xlabel('Predicted Probability', fontsize=14)
    ax_hist.set_ylabel('Frequency \n (Log Scale)', fontsize=14, ha='center')
    ax_hist.tick_params(axis='x', labelsize=8)
    ax_hist.tick_params(axis='y', labelsize=8)
    ax_hist.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_fname[:-4] + "hist.pdf", bbox_inches='tight')

def gen_relcurve_params(y_true_mlp, y_prob_mlp, y_true_xgb, y_prob_xgb, n_bins=10):
    """
    Compute reliability curve parameters for MLP and XGBoost without plotting.

    Returns calibration curve arrays used by the multi-panel reliability diagram
    in __main__. Separated from plot_reliability_diagram to allow reuse across
    multiple subplots without recreating figures.

    Args:
        y_true_mlp (array-like): True binary labels for MLP evaluation.
        y_prob_mlp (array-like): MLP predicted probabilities.
        y_true_xgb (array-like): True binary labels for XGBoost evaluation.
        y_prob_xgb (array-like): XGBoost predicted probabilities.
        n_bins (int):            Number of uniform-width bins. Default: 10.

    Returns:
        prob_true_mlp (np.ndarray): Observed frequencies per bin for MLP.
        prob_pred_mlp (np.ndarray): Mean predicted probabilities per bin for MLP.
        prob_true_xgb (np.ndarray): Observed frequencies per bin for XGBoost.
        prob_pred_xgb (np.ndarray): Mean predicted probabilities per bin for XGBoost.
        noskill_x (list): x-coordinates for the no-skill reference line.
        noskill_y (list): y-coordinates for the no-skill reference line.
    """
    prob_true_mlp, prob_pred_mlp = calibration_curve(y_true_mlp, y_prob_mlp, n_bins=n_bins, strategy='uniform')
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_true_xgb, np.clip(y_prob_xgb, a_min=0.00001, a_max=0.99945),
                                                     n_bins=n_bins, strategy='uniform')

    noskill_x = []
    noskill_y = []

    for x in np.arange(0, 1.10, 0.1):
        noskill_x.append(x)
        noskill_y.append(0.5 * (np.mean(y_true_mlp) + x))

    return prob_true_mlp, prob_pred_mlp, prob_true_xgb, prob_pred_xgb, noskill_x, noskill_y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='gen-predictive-curves',
                                     description='Generate reliability diagrams and AUC plots')
    parser.add_argument('--mlp', required=True, help='Path to MLP predictions CSV')
    parser.add_argument('--xgb', required=True, help='Path to XGBoost predictions CSV')
    args = parser.parse_args()

    mlp_results = pd.read_csv(args.mlp)
    xgb_results = pd.read_csv(args.xgb)
    xgb_results.dropna(inplace=True)
    print(len(xgb_results), len(mlp_results))
    hist_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig = uplt.figure(pad=2, refwidth=3, sharex=True, sharey=False )
    axs = fig.subplots(nrows=2, ncols=3, height_ratios=[4, 1],grid=False,hspace=0, refwidth=3)
    axs.format(abc=['d','e','f','','',''], abcloc='ur', labelsize=20,xlabel = "Predicted Probability",)

    cp = sns.color_palette("colorblind", n_colors=11)
    params = gen_relcurve_params(mlp_results['MESH_bool_next_30'], mlp_results['MLP_next_30'],
                             xgb_results['MESH_bool_next_30'], xgb_results['xgb_pred_next30'], n_bins=10)
    prob_true_mlp, prob_pred_mlp, prob_true_xgb, prob_pred_xgb, noskill_x, noskill_y = params

    # Plot the reliability diagram
    axs[0].plot(prob_pred_mlp, prob_true_mlp, marker='s', color=cp[1])
    axs[0].plot(prob_pred_xgb, prob_true_xgb, marker='s', color=cp[2])
    axs[0].format(ylabel="Fraction of Positives")

    axs[0].plot([0, 1], [0, 1], color='k')
    axs[0].grid(True)
    # Add a legend
    n_bins=10
    axs[0].legend(handles=[ mpatches.Patch(color=cp[1], label='MLP'),
                        mpatches.Patch(color=cp[2], label='XGBoost')], fontsize=10, loc='upper left', ncols=1)
    #Histogram
    hist, bin_edges = np.histogram(mlp_results['MLP_next_30'], bins=hist_bins, range=None, density=None, weights=None)
    print(hist, bin_edges)
    axs[3].bar(bin_edges[0:-1] + 0.05, np.log10(hist), edgecolor='black', alpha=0.5, color=cp[1])
    histx, bin_edgesx = np.histogram(xgb_results['xgb_pred_next30'], bins=bin_edges, range=None, density=None, weights=None)
    axs[3].bar(bin_edgesx[0:-1] + 0.05, np.log10(histx), edgecolor='black', alpha=0.5, color=cp[2])
    axs[3].set_xlabel('Predicted Probability', fontsize=14)
    axs[3].set_ylabel('Frequency \n (Log Scale)', fontsize=14, ha='center')

    axs[3].tick_params(axis='x', labelsize=8)
    axs[3].tick_params(axis='y', labelsize=8)
    axs[3].grid(True, axis='y')


    #15-45
    params = gen_relcurve_params(mlp_results['MESH_bool_15_45'], mlp_results['MLP_15_45'],
                             xgb_results['MESH_bool_15_45'], xgb_results['xgb_pred_15_45'], n_bins=10)
    prob_true_mlp, prob_pred_mlp, prob_true_xgb, prob_pred_xgb, noskill_x, noskill_y = params
    # Plot the reliability diagram
    axs[1].plot(prob_pred_mlp, prob_true_mlp, marker='s', color=cp[1])
    axs[1].plot(prob_pred_xgb, prob_true_xgb, marker='s', color=cp[2])
    axs[1].plot([0, 1], [0, 1], color='k')
    # Add a legend
    axs[1].legend(handles=[mpatches.Patch(color=cp[1], label='MLP'),
                           mpatches.Patch(color=cp[2], label='XGBoost')], fontsize=10, loc='upper left', ncols=1)
    axs[1].grid(True)
    # Histogram
    hist, bin_edges = np.histogram(mlp_results['MLP_15_45'], bins=hist_bins, range=None, density=None, weights=None)
    print(hist, bin_edges)
    axs[4].bar(bin_edges[0:-1]+ 0.05, np.log10(hist), edgecolor='black', alpha=0.5, color=cp[1])
    histx, bin_edgesx = np.histogram(xgb_results['xgb_pred_15_45'], bins=bin_edges, range=None, density=None, weights=None)
    axs[4].bar(bin_edgesx[0:-1]+ 0.05, np.log10(histx), edgecolor='black', alpha=0.5, color=cp[2])
    axs[4].tick_params(axis='x', labelsize=8)
    axs[4].tick_params(axis='y', labelsize=8)
    axs[4].grid(True, axis='y')
    #
    # # 30 - 60
    params = gen_relcurve_params(mlp_results['MESH_bool_last_30'], mlp_results['MLP_last_30'],
                                 xgb_results['MESH_bool_last_30'], xgb_results['xgb_pred_last30'], n_bins=10)
    prob_true_mlp, prob_pred_mlp, prob_true_xgb, prob_pred_xgb, noskill_x, noskill_y = params
    # Plot the reliability diagram
    axs[2].plot(prob_pred_mlp, prob_true_mlp, marker='s', color=cp[1])
    axs[2].plot(prob_pred_xgb, prob_true_xgb, marker='s', color=cp[2])
    axs[2].plot([0, 1], [0, 1], color='k')
    axs[2].grid(True)

    # Add a legend
    axs[2].legend(handles=[mpatches.Patch(color=cp[1], label='MLP'),
                           mpatches.Patch(color=cp[2], label='XGBoost')], fontsize=10, loc='upper left', ncols=1)
    # Histogram
    hist, bin_edges = np.histogram(mlp_results['MLP_last_30'], bins=hist_bins, range=None, density=None, weights=None)
    print(hist, bin_edges)
    axs[5].bar(bin_edges[0:-1]+ 0.05, np.log10(hist), edgecolor='black', alpha=0.5, color=cp[1])
    histx, bin_edgesx = np.histogram(xgb_results['xgb_pred_last30'], bins=bin_edges,
                                     range=None, density=None, weights=None)
    axs[5].bar(bin_edgesx[0:-1]+ 0.05, np.log10(histx), edgecolor='black', alpha=0.5, color=cp[2])
    axs[5].tick_params(axis='x', labelsize=8)
    axs[5].tick_params(axis='y', labelsize=8)
    axs[5].grid(True, axis='y')

    plt.savefig('results_figs/reliabilitydiagrams.pdf')


    """" AUC plots"""
    metrics = gen_auc_metrics(mlp_results['MESH_bool_next_30'], mlp_results['MLP_next_30'],
                                xgb_results['MESH_bool_next_30'], xgb_results['xgb_pred_next30'])
    (mlp_auc, fpr_m_points, tpr_m_points, xgb_auc, fpr_x_points, tpr_x_points) = metrics
    cp = sns.color_palette("colorblind", n_colors=11)
    fig = uplt.figure(sharey=True,pad=2)
    axs = fig.subplots(nrows=1, ncols=3)
    axs.format(abc=True, abcloc="ul", ylabel="True Positive Rate", xlabel="False Positive Rate", facecolor="white", labelsize=16)
    axs[0].plot(fpr_m_points, tpr_m_points, marker='o', markersize =5, linestyle='-', color=cp[1], label='MLP')
    axs[0].plot(fpr_x_points, tpr_x_points, marker='o', markersize =5, linestyle='-', color=cp[2], label='XGBoost')
    axs[0].plot([0, 1], [0, 1.0], color='k', linestyle='-')
    axs[0].legend(handles=[ mpatches.Patch(color=cp[1], label=f'MLP, AUC = {mlp_auc:.3f}'),
                            mpatches.Patch(color=cp[2], label=f'XGBoost, AUC = {xgb_auc:.3f}')],
                  fontsize=10, bbox_to_anchor=(0.2, 0.25), ncols=1)
    axs[0].format(title="0 - 30 min", titlesize=16)
    """ 15_45"""
    metrics = gen_auc_metrics(mlp_results['MESH_bool_15_45'], mlp_results['MLP_15_45'],
                              xgb_results['MESH_bool_15_45'], xgb_results['xgb_pred_15_45'])
    (mlp_auc, fpr_m_points, tpr_m_points, xgb_auc, fpr_x_points, tpr_x_points) = metrics
    axs[1].plot(fpr_m_points, tpr_m_points, marker='o', markersize=5, linestyle='-', color=cp[1], label='MLP')
    axs[1].plot(fpr_x_points, tpr_x_points, marker='o', markersize=5, linestyle='-', color=cp[2], label='XGBoost')
    axs[1].plot([0, 1], [0, 1.0], color='k', linestyle='-')
    axs[1].legend(handles=[mpatches.Patch(color=cp[1], label=f'MLP, AUC = {mlp_auc:.3f}'),
                           mpatches.Patch(color=cp[2], label=f'XGBoost AUC = {xgb_auc:.3f}')],
                  fontsize=10, bbox_to_anchor=(0.2, 0.25), ncols=1)
    axs[1].format(title="15 - 45 min", titlesize=16)
    """ 30-50"""
    metrics = gen_auc_metrics(mlp_results['MESH_bool_last_30'], mlp_results['MLP_last_30'],
                              xgb_results['MESH_bool_last_30'], xgb_results['xgb_pred_last30'])
    (mlp_auc, fpr_m_points, tpr_m_points, xgb_auc, fpr_x_points, tpr_x_points) = metrics
    axs[2].plot(fpr_m_points, tpr_m_points, marker='o', markersize=5, linestyle='-', color=cp[1], label='MLP')
    axs[2].plot(fpr_x_points, tpr_x_points, marker='o', markersize=5, linestyle='-', color=cp[2], label='XGBoost')
    axs[2].plot([0, 1], [0, 1.0], color='k', linestyle='-')
    axs[2].legend(handles=[mpatches.Patch(color=cp[1], label=f'MLP, AUC = {mlp_auc:.3f}'),
                           mpatches.Patch(color=cp[2], label=f'XGBoost, AUC = {xgb_auc:.3f}')],
                  fontsize=10, ncols=1, bbox_to_anchor=(0.2, 0.25), )
    axs[2].format(title="30 - 60 min", titlesize=16)
    # plt.show()
    fig.save("results_figs/AUC_plots.pdf")