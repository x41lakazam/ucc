import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os
from pathlib import Path
from scipy import stats
from typing import List, Dict, Optional

# --- Logger Setup ---
# Configure logger for consistent output format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


class Config:
    """
    Groups all configuration variables and file paths for easy management.
    """
    # --- Input Data Paths ---
    CSV_PATH_CLUSTER_A = '/mtrsysgwork/yyacobovich/install/all2allv/raw_latency_res/combined_latency_new_cc_16N.csv'
    CSV_PATH_CLUSTER_B = '/mtrsysgwork/yyacobovich/install/all2allv/raw_latency_res/combined_latency_16N_120625_oci.csv'
    MATRIX_FILES_PATH = '/mtrsysgwork/yyacobovich/install/all2allv/transfer_matrices/12_06_25/16N'

    # --- Output Directories ---
    BASE_OUTPUT_PATH = '/mtrsysgwork/yyacobovich/install/all2allv/graphs4/new_cc/16N'
    ORIGINAL_GRAPHS_PATH = os.path.join(BASE_OUTPUT_PATH, 'original_graphs')
    AGGREGATED_GRAPHS_PATH = os.path.join(BASE_OUTPUT_PATH, 'aggregated_latency_histograms')
    
    # --- Output File Names ---
    JCT_CSV_PATH = os.path.join(BASE_OUTPUT_PATH, 'jct_results.csv')
    ENRICHED_DATA_CSV_PATH = os.path.join(ORIGINAL_GRAPHS_PATH, 'enriched_average_latencies.csv')
    NORMALITY_TEST_CSV_PATH = os.path.join(ORIGINAL_GRAPHS_PATH, 'normality_test_results.csv')

    # --- Benchmark Parameters ---
    NUM_RANKS = 16
    BENCHMARKS = ['0', '1', '2', '3']
    EXISTING_MATRICES = ['0', '1', '2']
    
    # --- Plotting & Naming ---
    MATRIX_NAMES = {
        "0_0": "Token choice, Iteration 0",
        "0_1": "Token choice, Iteration 1",
        "0_2": "Token choice, Iteration 2",
        "1_0": "Token choice bias, Iteration 0",
        "1_1": "Token choice bias, Iteration 1",
        "1_2": "Token choice bias, Iteration 2",
        "2_0": "Token choice, random #experts, Iteration 0",
        "2_1": "Token choice, random #experts, Iteration 1",
        "2_2": "Token choice, random #experts, Iteration 2",
        "3_0": "Token choice, random #experts, random token size, Iteration 0",
        "3_1": "Token choice, random #experts, random token size, Iteration 1",
        "3_2": "Token choice, random #experts, random token size, Iteration 2"
    }


# --- Helper & Data Processing Functions ---

def parse_size_to_bytes(size_str: str) -> int:
    """
    Converts a formatted string with a unit (e.g., '10k', '2M') to bytes.
    If no unit is found, assumes Kilobytes (K) as default.
    """
    size_str = size_str.lower().strip()
    if not size_str:
        return 0
    
    units = {'g': 1024**3, 'm': 1024**2, 'k': 1024, 'b': 1}
    unit = 1024  # Default to Kilobytes
    value_str = size_str

    if size_str[-1] in units:
        unit = units[size_str[-1]]
        value_str = size_str[:-1]
    
    try:
        value = float(value_str)
    except ValueError:
        log.error(f"Could not parse value from size string: '{size_str}'")
        return 0
        
    return int(unit * value)


def load_matrix(matrix_file: Path) -> np.ndarray:
    """Loads a transfer matrix from a file into a numpy array."""
    try:
        with open(matrix_file, "r") as f:
            rows = [[parse_size_to_bytes(x) for x in line.strip().split()] for line in f if line.strip()]
        return np.array(rows)
    except FileNotFoundError:
        log.warning(f"Matrix file not found: {matrix_file}")
        return np.array([])


def calculate_and_save_jct(df: pd.DataFrame, output_path: str) -> pd.Series:
    """
    Calculates the average Job Completion Time (JCT) for each matrix.
    JCT is the max latency across all ranks per iteration, averaged over all iterations.
    """
    log.info("Calculating Average Job Completion Time (JCT) per matrix.")
    df['iteration'] = df.groupby(['matrix', 'rank']).cumcount()
    jct_per_iteration = df.groupby(['matrix', 'iteration'])['latency'].max()
    avg_jct_per_matrix = jct_per_iteration.groupby('matrix').mean()
    
    avg_jct_per_matrix.to_csv(output_path, index=True)
    log.info(f"Average JCT results saved to {output_path}")
    
    return avg_jct_per_matrix


def enrich_data_with_metrics(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Enriches the DataFrame with message size and effective bandwidth data.
    """
    log.info("Enriching data with message size and effective bandwidth.")
    avg_df = df.groupby(["matrix", "rank"])["latency"].mean().reset_index()
    avg_df['msg_size_sent'] = np.nan
    avg_df['msg_size_recv'] = np.nan
    avg_df['effective_bw_mbps'] = np.nan

    for benchmark in config.BENCHMARKS:
        for mat_ix in config.EXISTING_MATRICES:
            matrix_id = f"{benchmark}_{mat_ix}"
            matrix_file = Path(config.MATRIX_FILES_PATH) / f"bench{benchmark}_iter{mat_ix}" / "0"
            
            mat = load_matrix(matrix_file)
            if mat.size == 0:
                continue

            # Pad matrix if its dimensions are smaller than NUM_RANKS
            if mat.shape[0] < config.NUM_RANKS or mat.shape[1] < config.NUM_RANKS:
                padded_mat = np.zeros((config.NUM_RANKS, config.NUM_RANKS))
                padded_mat[:mat.shape[0], :mat.shape[1]] = mat
                mat = padded_mat

            for rank in range(config.NUM_RANKS):
                mask = (avg_df['matrix'] == matrix_id) & (avg_df['rank'] == rank)
                if not mask.any():
                    continue
                
                # Calculate total sent and received bytes for the rank
                sent_bytes = np.sum(mat[rank, :])
                recv_bytes = np.sum(mat[:, rank])
                avg_df.loc[mask, 'msg_size_sent'] = sent_bytes
                avg_df.loc[mask, 'msg_size_recv'] = recv_bytes
                
                # Calculate effective bandwidth
                latency_us = avg_df.loc[mask, 'latency'].iloc[0]
                if latency_us > 0:
                    latency_s = latency_us / 1_000_000
                    # Bandwidth is calculated based on the larger of sent/received data
                    max_msg_bytes = max(sent_bytes, recv_bytes)
                    bw_mbps = (max_msg_bytes / latency_s) / (1024 * 1024)
                    avg_df.loc[mask, 'effective_bw_mbps'] = bw_mbps
                    
    return avg_df


def perform_normality_tests(df: pd.DataFrame, config: Config):
    """
    Performs Kolmogorov-Smirnov normality tests on aggregated latency distributions.
    and also performs shapiro-wilk test
    """
    log.info("Calculating normality test statistics for aggregated latencies.")
    df['iteration'] = df.groupby(['matrix', 'rank']).cumcount()
    agg_latencies = df.groupby(['matrix', 'iteration'])['latency'].agg(['max', 'min', 'median']).reset_index()
    agg_latencies.rename(columns={'max': 'max_latency', 'min': 'min_latency', 'median': 'median_latency'}, inplace=True)

    results = []
    metrics = ['max_latency', 'min_latency', 'median_latency']
    
    for matrix_id in agg_latencies['matrix'].unique():
        for metric in metrics:
            data = agg_latencies[agg_latencies['matrix'] == matrix_id][metric]
            
            # Standardize data for comparison against a standard normal distribution
            if data.std() > 0:
                standardized_data = (data - data.mean()) / data.std()
                ks_statistic, p_value = stats.kstest(standardized_data, 'norm')
                sw_statistic, sw_p_value = stats.shapiro(data)
            else:
                # A distribution with no variance is not normal
                ks_statistic, p_value = np.inf, 0.0
                sw_statistic, sw_p_value = np.inf, 0.0
            matrix_name = config.MATRIX_NAMES.get(matrix_id, f"Matrix {matrix_id}")
            results.append({
                "Matrix Name": matrix_name,
                "Metric": metric,
                "KS-Statistic (Distance)": ks_statistic,
                "P-Value": p_value,
                "SW-Statistic": sw_statistic,
                "SW-P-Value": sw_p_value
            })
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.NORMALITY_TEST_CSV_PATH, index=False)
    log.info(f"Normality test results saved to: {config.NORMALITY_TEST_CSV_PATH}")
    log.info(f"\n{results_df.to_string()}")
    return results_df


# --- Plotting Functions ---

def plot_latency_boxplot(df: pd.DataFrame, config: Config):
    """Generates and saves a boxplot comparing latency distributions across all matrices."""
    log.info("Generating latency comparison boxplot.")
    num_matrices = len(df['matrix'].unique())
    fig_width = max(12, num_matrices * 1.5)
    
    plt.figure(figsize=(fig_width, 8))
    sns.boxplot(x='matrix', y='latency', data=df, palette='viridis', showfliers=False)
    
    plt.title('Latency Distribution Comparison Across Matrices', fontsize=18, pad=20)
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('Latency (us)', fontsize=14)
    
    # Use mapped names for x-axis labels
    matrix_labels = [config.MATRIX_NAMES.get(str(x.get_text()), x.get_text()) for x in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(matrix_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(config.ORIGINAL_GRAPHS_PATH, 'boxplot_latency_comparison.png')
    plt.savefig(save_path)
    plt.close()


def plot_aggregated_histograms(df: pd.DataFrame, config: Config):
    """Calculates and plots histograms of aggregated (min, max, median) latencies."""
    log.info("Generating histograms of aggregated latencies per iteration.")
    df['iteration'] = df.groupby(['matrix', 'rank']).cumcount()
    agg_latencies = df.groupby(['matrix', 'iteration'])['latency'].agg(['max', 'min', 'median']).reset_index()
    agg_latencies.rename(columns={'max': 'max_latency', 'min': 'min_latency', 'median': 'median_latency'}, inplace=True)
    
    metrics = {
        'max_latency': 'Max Latency Across Ranks (us)',
        'min_latency': 'Min Latency Across Ranks (us)',
        'median_latency': 'Median Latency Across Ranks (us)'
    }

    for matrix_id in agg_latencies['matrix'].unique():
        matrix_name = config.MATRIX_NAMES.get(matrix_id, matrix_id)
        matrix_data = agg_latencies[agg_latencies['matrix'] == matrix_id]
        
        for col, label in metrics.items():
            plt.figure(figsize=(12, 7))
            plt.hist(matrix_data[col], bins=50, edgecolor='black', color='coral')
            plt.title(f'Distribution of {label.split("(")[0].strip()} for {matrix_name}\n(Each data point is one iteration)')
            plt.xlabel(label)
            plt.ylabel('Frequency (Count of Iterations)')
            plt.grid(axis='y', alpha=0.75)
            
            save_path = os.path.join(config.AGGREGATED_GRAPHS_PATH, f'matrix_{matrix_id}_{col}_histogram.png')
            plt.savefig(save_path)
            plt.close()


def plot_size_vs_latency_dual_axis(enriched_df: pd.DataFrame, config: Config):
    """Creates dual-axis bar charts comparing message size and latency per rank."""
    log.info("Generating dual-axis bar plots (Message Size vs. Latency).")
    bytes_to_mb = 1024 * 1024
    
    for matrix_id, matrix_data in enriched_df.groupby('matrix'):
        if matrix_data.empty or matrix_data['msg_size_sent'].isnull().all():
            continue
        
        matrix_name = config.MATRIX_NAMES.get(matrix_id, matrix_id)
        matrix_data = matrix_data.sort_values(by='rank')
        
        ranks = matrix_data['rank']
        msg_sizes_mb = matrix_data['msg_size_sent'] / bytes_to_mb
        latencies = matrix_data['latency']

        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = ax1.twinx()
        bar_width = 0.4
        indices = np.arange(len(ranks))
        
        ax1.bar(indices, msg_sizes_mb, width=bar_width, color='royalblue', label='Msg Size Sent')
        ax1.set_xlabel('Rank', fontsize=14)
        ax1.set_ylabel('Msg Size Sent (MB)', color='royalblue', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_ylim(top=max(msg_sizes_mb) * 1.2)

        ax2.bar(indices + bar_width, latencies, width=bar_width, color='darkorange', label='Latency')
        ax2.set_ylabel('Average Latency (us)', color='darkorange', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(top=max(latencies) * 1.2)
        
        plt.xticks(indices + bar_width / 2, ranks)
        plt.title(f'Sent Message Size vs. Average Latency ({matrix_name})', fontsize=18, pad=20)
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        fig.tight_layout()
        
        save_path = os.path.join(config.ORIGINAL_GRAPHS_PATH, f'matrix{matrix_id}_sent_vs_latency.png')
        plt.savefig(save_path)
        plt.close()


def plot_bandwidth_per_rank(enriched_df: pd.DataFrame, config: Config):
    """Creates bar charts showing effective bandwidth per rank for each matrix."""
    log.info("Generating single-bar plots for effective bandwidth.")
    for matrix_id, matrix_data in enriched_df.groupby('matrix'):
        if matrix_data.empty or matrix_data['effective_bw_mbps'].isnull().all():
            log.warning(f"Skipping bandwidth plot for matrix {matrix_id} due to missing data.")
            continue
            
        matrix_name = config.MATRIX_NAMES.get(matrix_id, matrix_id)
        matrix_data = matrix_data.sort_values(by='rank')
        
        plt.figure(figsize=(20, 10))
        plt.bar(matrix_data['rank'], matrix_data['effective_bw_mbps'], color='forestgreen')
        
        plt.xlabel('Rank', fontsize=14)
        plt.ylabel('Effective Bandwidth (MB/s)', fontsize=14)
        plt.title(f'Effective Bandwidth per Rank ({matrix_name})', fontsize=18, pad=20)
        
        if not matrix_data['effective_bw_mbps'].empty:
            plt.ylim(top=matrix_data['effective_bw_mbps'].max() * 1.2)
            
        plt.xticks(matrix_data['rank'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(config.ORIGINAL_GRAPHS_PATH, f'matrix{matrix_id}_bandwidth.png')
        plt.savefig(save_path)
        plt.close()


def plot_comparison_bar_chart(
    df: pd.DataFrame, col1: str, col2: str, label1: str, label2: str, ylabel: str, title: str, save_path: str
):
    """Creates a generic side-by-side bar chart to compare two sets of data."""
    log.info(f"Generating comparison bar chart: {title}")
    
    indices = np.arange(len(df.index))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(20, 10))
    
    rects1 = ax.bar(indices - bar_width/2, df[col1], bar_width, label=label1, color='#4ea72f')
    rects2 = ax.bar(indices + bar_width/2, df[col2], bar_width, label=label2, color='darkgrey')

    ax.set_xlabel('Matrix', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xticks(indices)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend(fontsize=12)
    
    # Set y-axis limit and add bar labels
    max_val = max(df[col1].max(), df[col2].max())
    if max_val > 0:
        ax.set_ylim(top=max_val * 1.2)
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Comparison chart saved to {save_path}")


def plot_comparison_dist_lines(
    agg_df1: pd.DataFrame, agg_df2: pd.DataFrame, metric: str, 
    name1: str, name2: str, config: Config, output_path: str
):
    """Compares the KDE for a specific aggregated latency metric from two clusters."""
    log.info(f"Generating comparison distribution plot for: {metric}")
    
    plt.figure(figsize=(20, 12))
    
    unique_matrices = sorted(list(set(agg_df1['matrix'].unique()) | set(agg_df2['matrix'].unique())))
    palette = sns.color_palette("husl", len(unique_matrices))
    matrix_color_map = {matrix_id: color for matrix_id, color in zip(unique_matrices, palette)}

    max_x = 0
    for matrix_id in unique_matrices:
        matrix_name = config.MATRIX_NAMES.get(matrix_id, matrix_id)
        
        # Plot for Cluster 1 (solid line)
        data1 = agg_df1[agg_df1['matrix'] == matrix_id][metric].dropna()
        if not data1.empty and data1.std() > 0:
            sns.kdeplot(data1, label=f'{name1} - {matrix_name}', color=matrix_color_map[matrix_id], linestyle='-', lw=2.5)
            max_x = max(max_x, data1.quantile(0.99))

        # Plot for Cluster 2 (dashed line)
        data2 = agg_df2[agg_df2['matrix'] == matrix_id][metric].dropna()
        if not data2.empty and data2.std() > 0:
            sns.kdeplot(data2, label=f'{name2} - {matrix_name}', color=matrix_color_map[matrix_id], linestyle='--', lw=2.5)
            max_x = max(max_x, data2.quantile(0.99))

    plt.title(f'Comparison of {metric} Distributions ({name1} vs. {name2})', fontsize=20, pad=20)
    plt.xlabel('Latency (us)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Cluster - Matrix', loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, max_x)
    plt.tight_layout()
    
    save_path = os.path.join(output_path, f"cluster_comparison_{metric.replace(' ', '_').lower()}_dists.png")
    plt.savefig(save_path)
    plt.close()
    log.info(f"Comparison distribution plot saved to {save_path}")

# --- Main Execution Flow ---

def load_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Loads data from a CSV file, handling errors."""
    try:
        df = pd.read_csv(csv_path)
        log.info(f"Successfully loaded data from: {csv_path}")
        return df
    except FileNotFoundError:
        log.error(f"FATAL: CSV file not found at {csv_path}. Cannot proceed with this data.")
        return None


def run_single_cluster_analysis(df: pd.DataFrame, config: Config):
    """Runs all analyses and generates all plots for a single cluster."""
    log.info("--- Starting Single-Cluster Analysis ---")
    
    # Calculate and display JCT
    jct_results = calculate_and_save_jct(df.copy(), config.JCT_CSV_PATH)
    log.info("Average JCT (us):")
    for matrix_id, avg_jct in jct_results.items():
        matrix_name = config.MATRIX_NAMES.get(matrix_id, matrix_id)
        log.info(f"  {matrix_name}: {avg_jct:.2f} us")

    # Generate basic latency plots
    plot_latency_histograms(df, config)
    plot_latency_boxplot(df, config)
    
    # Enrich data and generate advanced plots
    enriched_df = enrich_data_with_metrics(df.copy(), config)
    enriched_df.to_csv(config.ENRICHED_DATA_CSV_PATH, index=False)
    log.info(f"Saved enriched data to {config.ENRICHED_DATA_CSV_PATH}")
    
    plot_size_vs_latency_dual_axis(enriched_df, config)
    plot_bandwidth_per_rank(enriched_df, config)

    # Analyze aggregated distributions
    plot_aggregated_histograms(df.copy(), config)
    perform_normality_tests(df.copy(), config)
    
    log.info("--- Single-Cluster Analysis Finished ---")


def run_comparison_analysis(df1: pd.DataFrame, df2: pd.DataFrame, config: Config):
    """Runs all comparison analyses and generates plots for two clusters."""
    log.info("--- Starting Two-Cluster Comparison Analysis ---")
    
    # Enrich both dataframes to get bandwidth metrics for comparison
    enriched_df1 = enrich_data_with_metrics(df1.copy(), config)
    enriched_df2 = enrich_data_with_metrics(df2.copy(), config)

    # 1. Compare Mean Bandwidth
    mean_bw1 = enriched_df1.groupby('matrix')['effective_bw_mbps'].mean()
    mean_bw2 = enriched_df2.groupby('matrix')['effective_bw_mbps'].mean()
    mean_comp_df = pd.DataFrame({'IL1_Mean_BW': mean_bw1, 'OCI-NRT_Mean_BW': mean_bw2}).dropna()
    
    if not mean_comp_df.empty:
        mean_comp_df.index = mean_comp_df.index.map(config.MATRIX_NAMES)
        plot_comparison_bar_chart(
            df=mean_comp_df, col1='IL1_Mean_BW', col2='OCI-NRT_Mean_BW',
            label1='IL1', label2='OCI-NRT',
            ylabel='Mean Effective BW (MB/s)',
            title='Mean Bandwidth Comparison Between Clusters',
            save_path=os.path.join(config.ORIGINAL_GRAPHS_PATH, 'cluster_comparison_mean_bw.png')
        )
    else:
        log.warning("Could not generate Mean BW comparison chart due to no overlapping matrix data.")

    # 2. Compare Bandwidth Variance
    var_bw1 = enriched_df1.groupby('matrix')['effective_bw_mbps'].var()
    var_bw2 = enriched_df2.groupby('matrix')['effective_bw_mbps'].var()
    var_comp_df = pd.DataFrame({'IL1_Var_BW': var_bw1, 'OCI-NRT_Var_BW': var_bw2}).dropna()
    
    if not var_comp_df.empty:
        var_comp_df.index = var_comp_df.index.map(config.MATRIX_NAMES)
        plot_comparison_bar_chart(
            df=var_comp_df, col1='IL1_Var_BW', col2='OCI-NRT_Var_BW',
            label1='IL1', label2='OCI-NRT',
            ylabel='Variance of Effective BW',
            title='Bandwidth Variance Comparison Between Clusters',
            save_path=os.path.join(config.ORIGINAL_GRAPHS_PATH, 'cluster_comparison_variance_bw.png')
        )
    else:
        log.warning("Could not generate BW Variance comparison chart due to no overlapping matrix data.")
    # 3. compare coefficient of variance (std/mean)
    std_bw1 = enriched_df1.groupby('matrix')['effective_bw_mbps'].std()
    std_bw2 = enriched_df2.groupby('matrix')['effective_bw_mbps'].std()
    mean_bw1 = enriched_df1.groupby('matrix')['effective_bw_mbps'].mean()
    mean_bw2 = enriched_df2.groupby('matrix')['effective_bw_mbps'].mean()
    std_comp_df = pd.DataFrame({'IL1_Std_BW': std_bw1, 'OCI-NRT_Std_BW': std_bw2}).dropna()
    std_comp_df['IL1_CoV_BW'] = std_comp_df['IL1_Std_BW'] / mean_bw1
    std_comp_df['OCI-NRT_CoV_BW'] = std_comp_df['OCI-NRT_Std_BW'] / mean_bw2
    
    if not std_comp_df.empty:
        std_comp_df.index = std_comp_df.index.map(config.MATRIX_NAMES)
        plot_comparison_bar_chart(
            df=std_comp_df, col1='IL1_CoV_BW', col2='OCI-NRT_CoV_BW',
            label1='IL1', label2='OCI-NRT',
            ylabel='Coefficient of Variance of Effective BW',
            title='Coefficient of Variance of Effective BW Comparison Between Clusters',
            save_path=os.path.join(config.ORIGINAL_GRAPHS_PATH, 'cluster_comparison_cov_bw.png')
        )
    else:
        log.warning("Could not generate CoV BW comparison chart due to no overlapping matrix data.")
    # 4. Compare Aggregated Latency Distributions   
    df1['iteration'] = df1.groupby(['matrix', 'rank']).cumcount()
    agg1 = df1.groupby(['matrix', 'iteration'])['latency'].agg(['max', 'min', 'median']).reset_index()
    agg1.columns = ['matrix', 'iteration', 'Max Latency', 'Min Latency', 'Median Latency']
    
    df2['iteration'] = df2.groupby(['matrix', 'rank']).cumcount()
    agg2 = df2.groupby(['matrix', 'iteration'])['latency'].agg(['max', 'min', 'median']).reset_index()
    agg2.columns = ['matrix', 'iteration', 'Max Latency', 'Min Latency', 'Median Latency']

    for metric in ['Max Latency', 'Min Latency', 'Median Latency']:
        plot_comparison_dist_lines(
            agg_df1=agg1, agg_df2=agg2, metric=metric,
            name1='IL1', name2='OCI-NRT',
            config=config, output_path=config.ORIGINAL_GRAPHS_PATH
        )
        
    # 5. calculate normality test results
    normality_test_results_1 = perform_normality_tests(df1, config)
    normality_test_results_2 = perform_normality_tests(df2, config)
    # concatenate the normality test results with the matrix names
    normality_test_results_1['Cluster'] = 'IL1'
    normality_test_results_2['Cluster'] = 'OCI-NRT'
    normality_test_results = pd.concat([normality_test_results_1, normality_test_results_2])
    # save the normality test results
    normality_test_results.to_csv(os.path.join(config.ORIGINAL_GRAPHS_PATH, 'normality_test_results_both_clusters.csv'), index=False)
    log.info("--- Two-Cluster Comparison Analysis Finished ---")


def main():
    """Main function to orchestrate the data analysis and plotting workflow."""
    config = Config()
    
    # Create necessary output directories
    os.makedirs(config.ORIGINAL_GRAPHS_PATH, exist_ok=True)
    os.makedirs(config.AGGREGATED_GRAPHS_PATH, exist_ok=True)

    # Load data for both clusters
    df_cluster_a = load_data(config.CSV_PATH_CLUSTER_A)
    df_cluster_b = load_data(config.CSV_PATH_CLUSTER_B)

    # Run analysis for the first cluster (Cluster A)
    if df_cluster_a is not None:
        run_single_cluster_analysis(df_cluster_a, config)

    # Run comparison analysis only if both data files were loaded successfully
    if df_cluster_a is not None and df_cluster_b is not None:
        run_comparison_analysis(df_cluster_a, df_cluster_b, config)
    else:
        log.warning("Skipping two-cluster comparison because one or both data files could not be loaded.")

    log.info("Script finished successfully.")


if __name__ == "__main__":
    main()