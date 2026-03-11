import os
import pandas as pd
from src.data_utils import df_sample_from_random_rows
from src.models import (
    red_neuronal_categorica,
    leave_one_out_kmeans_country,
    leave_one_out_red_neuronal_categorica
)
from src.evaluation import (
    compute_global_metrics,
    df_results_comparing_both_againts_PPI_and_GPI
)

def main():
    # 1. Configuration and Data Loading
    # ---------------------------------------------------------
    # Update this path to point to your actual cleaned dataset
    DATA_PATH = "data/clean_data.csv" 
    OUTPUT_DIR = "output"
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from: {DATA_PATH}")
    try:
        df_clean_data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Please ensure the file exists.")
        return

    # 2. Define the sample sizes to iterate over
    # ---------------------------------------------------------
    # This replaces the manual df_sample_50, df_sample_100, etc.
    sample_sizes = [50, 100, 200, 400, 600, 800, 1000]
    random_state = 91
    
    # 3. Execution Loop
    # ---------------------------------------------------------
    for i, size in enumerate(sample_sizes):
        print(f"\n{'='*40}")
        print(f"Processing sample size: {size} (Iteration {i})")
        print(f"{'='*40}")
        
        try:
            # Generate the sample for the current size
            df_sample = df_sample_from_random_rows(df_clean_data, size, random_state=random_state)
            
            # Initialize the Neural Network model
            # Note: The input shape (384, 1) should match your embedding dimensions. 
            # ChromaDB's default is 384.
            model_nn = red_neuronal_categorica((384, 1))
            
            # Run Leave-One-Out for KMeans
            print("Running KMeans LOO...")
            result_kmeans = leave_one_out_kmeans_country(df_sample)
            
            # Run Leave-One-Out for Neural Network
            print("Running Neural Network LOO...")
            result_nn = leave_one_out_red_neuronal_categorica(df_sample, model_nn)
            
            # Compute global metrics
            print("Computing metrics...")
            metrics_km, cls_report_km, cm_km, metrics_df_km = compute_global_metrics(result_kmeans)
            metrics_nn, cls_report_nn, cm_nn, metrics_df_nn = compute_global_metrics(result_nn)
            
            # Merge results and compare against static indices (PPI)
            df_merge = df_results_comparing_both_againts_PPI_and_GPI(result_kmeans, result_nn)
            
            # 4. Save Outputs
            # ---------------------------------------------------------
            # Using 'size' in the filename makes it easier to track which file is which
            metrics_km_path = os.path.join(OUTPUT_DIR, f"metrics_df_kmeans_size_{size}.csv")
            metrics_nn_path = os.path.join(OUTPUT_DIR, f"metrics_df_nn_size_{size}.csv")
            merge_path = os.path.join(OUTPUT_DIR, f"df_merge_size_{size}.csv")
            
            metrics_df_km.to_csv(metrics_km_path, index=False)
            metrics_df_nn.to_csv(metrics_nn_path, index=False)
            df_merge.to_csv(merge_path, index=False)
            
            print(f"Successfully saved iteration {i} (size {size}) files to '{OUTPUT_DIR}/'")
            
        except Exception as e:
            print(f"An error occurred while processing sample size {size}: {e}")

if __name__ == "__main__":
    main()