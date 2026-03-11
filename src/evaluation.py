# src/evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    precision_recall_fscore_support, confusion_matrix, classification_report
)
from .constants import DICT_PEACE_PPI_2023, DICT_PEACE_PPI_2024

def compute_global_metrics(results_df):
    """
    Calculates global metrics on the Leave-One-Out predictions (country level).
    Returns (metrics_dict, classification_report_str, confusion_matrix_array)
    and a DataFrame version of the metrics for easy saving.
    """
    y_true = results_df["true_peace"].astype(int).values
    y_pred = results_df["predicted_peace"].astype(int).values

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    # Avoid warnings when a class does not appear in predictions
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cls_report = classification_report(y_true, y_pred, labels=[0,1], output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(acc, 6), "balanced_accuracy": round(bacc, 6),
        "precision_binary_1": round(p_bin, 6), "recall_binary_1": round(r_bin, 6),
        "f1_binary_1": round(f1_bin, 6), "precision_macro": round(p_macro, 6),
        "recall_macro": round(r_macro, 6), "f1_macro": round(f1_macro, 6),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        "n_samples": int(len(y_true)), "n_samples_pos": int(y_true.sum()),
        "n_samples_neg": int(len(y_true) - y_true.sum())
    }

    return metrics, cls_report, cm, pd.DataFrame([metrics])

def df_results_comparing_both_againts_PPI_and_GPI(df_metrics_kmeans, df_metrics_nn):
    """
    Merges KMeans and Neural Network results into a single DataFrame, 
    incorporating static PPI scores for comparison.
    """
    df_metrics_kmeans = df_metrics_kmeans.copy()
    df_metrics_nn = df_metrics_nn.copy()
    
    df_metrics_kmeans["model"] = "kmeans"
    df_metrics_nn["model"] = "nn"
    
    df_merge = pd.concat([df_metrics_kmeans, df_metrics_nn], ignore_index=True)
    df_merge["peace_class"] = np.where(df_merge["true_peace"] == 1, "pos", "neg")
    
    df_ppi_2023 = pd.DataFrame(list(DICT_PEACE_PPI_2023.items()), columns=["country", "peace_PPI_2023"])
    df_ppi_2024 = pd.DataFrame(list(DICT_PEACE_PPI_2024.items()), columns=["country", "peace_PPI_2024"])
    
    df_merge = pd.merge(df_merge, df_ppi_2023, on="country", how="left")
    df_merge = pd.merge(df_merge, df_ppi_2024, on="country", how="left")
    
    return df_merge