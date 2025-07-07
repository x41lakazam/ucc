# Alltoallv Performance Test with Transfer Matrices

## Background

In Mixture-of-Experts (MoE) models, each token is routed to a subset of experts. This dynamic assignment leads to irregular communication patterns across devices.

To support this efficiently, **`alltoallv`** is used—it allows each process to send and receive variable-sized data to/from every other process. This is critical for MoE training where:

- Different tokens require different experts.
- Expert assignment changes per batch.
- Communication volume varies across ranks.

`alltoallv` enables scalable, flexible token-to-expert routing and is essential for high-performance distributed MoE training.


---

## Overview

This document describes the implementation, configuration, and usage of the modified UCC perftest for evaluating `alltoallv` performance using custom transfer matrices. The test runs multiple iterations over a set of predefined data exchange patterns, providing detailed timing information and log outputs.

---

## Implementation Details

The performance test is structured around **main iterations**, each consisting of a complete cycle through all provided transfer matrices. This design allows consistent benchmarking across different communication patterns.

- The `run_single_coll_test` method handles the main iterations.
- An inner loop was added to run all `alltoallv` operations with the provided transfer matrices during each main iteration.
- All transfer matrices are loaded into memory before the test begins.
- A new `coll->pre_run` method was added to adjust collective arguments per matrix before each inner execution.
- Send/receive buffers are allocated once at the beginning using the largest matrix size for simplicity and performance.

> **Note:** Buffer allocation time is not measured. In real workloads (e.g., PyTorch), buffer registration may introduce additional overhead.

---

## Output Interpretation

### Console Output (stdout)

The output reports average, minimum, and maximum latency across all ranks for all transfer matrices:

- **Max time** is the key metric—it indicates the slowest rank to finish during each main iteration and reflects overall synchronization.
- Each matrix execution is followed by a **barrier**, ensuring that all ranks begin the next collective at the same time.
- Timing does **not** include the barrier itself.

### Inner Logs

Detailed per-rank latency data is logged to files specified by the `UCC_PT_COLL_INNER_LOG_FILE` environment variable:

- One `.log` file per rank.
- Each file contains a latency entry (in microseconds) for every main iteration.

---

## Parameters

### Transfer Matrices

- Each matrix file should contain rows of byte counts, space-separated.
- Units like `M` (megabytes) and `G` (gigabytes) are supported (e.g., `5M`, `1G`).
- All matrix files must reside in a single directory.
- Set the following environment variables:
  - `UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR`: path to matrix directory.
  - `UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT`: number of matrix files.

> File names should be numeric (e.g., `0`, `1`, …, `N-1`), where `N` is the count.

Mismatch between file count and `*_COUNT` will result in an error.

### UCC Perftest CLI Flags

- Set `-b 0 -e 0` (message size is controlled by matrices, not these flags).
- Set `-j` (number of inner iterations) to match `*_MATRICES_COUNT`.

---

## Environment Variables Summary

| Variable                                         | Description                                                       |
|--------------------------------------------------|-------------------------------------------------------------------|
| `UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT`  | Number of transfer matrices                                       |
| `UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR`    | Directory containing matrix files                                 |
| `UCC_PT_COLL_INNER_LOG_FILE`                     | Path prefix for per-rank inner log files                          |
| `UCC_TL_UCP_ALLTOALLV_PAIRWISE_NUM_POSTS`        | Max outstanding messages in pairwise algorithm                    |
| `UCC_TL_UCP_TUNE=alltoallv:pairwise`             | Enable tuning for pairwise algorithm                              |
| `UCC_TLS=ucp`                                     | Specifies UCP transport for the test                              |

> Additional UCX-related flags may be needed depending on cluster configuration. Refer to the [UCX GitHub repo](https://github.com/openucx/ucx) for details.

---

## Container Information

**Container URI:**
docker://gitlab-master.nvidia.com:5005/yyacobovich/my_containers:all2allv_ucc_perf_pytorch25_03_v1.4_inner_logs
It includes:
- PyTorch 25_03
- The specific UCC and UCX binaries
- Logging support for inner loop latency measurements

---

## Matrix Generator

A matrix generator tool is available here:  
🔗 https://gitlab-master.nvidia.com/e2e-arch-network/a2aV_analysis_misc_tools

---

## Extraction Script

To analyze inner log files:

📄 [inner_logs_extract.py](https://github.com/x41lakazam/ucc/blob/yael_updates/tools/perf/alltoallv_files/inner_logs_extract.py)

- **Input**: Directory containing all `.log` files  
- **Output**: Single `.csv` with per-iteration latency per rank

---
## Alltoallv Inner Log Analysis Graphs

This section explains the various graphs generated from the Alltoallv inner logs (after extraction in privious script), providing insights into latency, bandwidth, and message size distributions and comparisons across different benchmark configurations and clusters.

📄 [graph_from_inner_log.py](https://github.com/x41lakazam/ucc/blob/yael_updates/tools/perf/alltoallv_files/graph_from_inner_log.py)

### Single-Cluster Analysis Graphs

These graphs are generated for a single cluster's performance data.


#### 1. Latency Distribution Comparison Boxplot

* **File Name**: `boxplot_latency_comparison.png`
* **Description**: This boxplot provides a comparative view of latency distributions across all analyzed matrices. Each box represents a different matrix, showing the median, quartiles, and potential outliers (though outliers are suppressed in these plots for clarity).
* **Insight**: Allows for quick visual comparison of central tendency and spread of latencies between different communication patterns.

#### 2. Aggregated Latency Histograms

* **File Name Pattern**: `matrix_{matrix_id}_{metric}_histogram.png` (e.g., `matrix_0_0_max_latency_histogram.png`)
* **Description**: These histograms show the distribution of aggregated latency metrics (Maximum, Minimum, and Median latency) across all ranks for each iteration within a given matrix. Each data point in these histograms represents the calculated metric (max, min, or median) from a single iteration of the benchmark.
* **Insight**: Provides insight into the consistency and variability of overall iteration performance rather than individual rank-to-rank latencies. For example, the "Max Latency" histogram shows the distribution of job completion times per iteration.

#### 3. Sent Message Size vs. Average Latency Dual-Axis Bar Charts

* **File Name Pattern**: `matrix{matrix_id}_sent_vs_latency.png`
* **Description**: These dual-axis bar charts compare the total message size sent by each rank with its average latency for a specific matrix. One y-axis represents message size (MB), and the other represents average latency (us).
* **Insight**: Helps to visualize the relationship between the amount of data a rank sends and the average latency it experiences, potentially highlighting bottlenecks or correlations.

#### 4. Effective Bandwidth Per Rank Bar Charts

* **File Name Pattern**: `matrix{matrix_id}_bandwidth.png`
* **Description**: These bar charts illustrate the calculated effective bandwidth (MB/s) for each rank within a given matrix. The effective bandwidth is derived from the maximum of sent/received bytes and the average latency.
* **Insight**: Shows how bandwidth is distributed among ranks and helps identify if any specific rank is underperforming in terms of data transfer efficiency.

#### 5. Normality Test Results

* **File Name**: `normality_test_results.csv`
* **Description**: This CSV file contains the results of Kolmogorov-Smirnov (KS) and Shapiro-Wilk (SW) normality tests performed on the aggregated latency distributions (max, min, median latencies per iteration).
* **Insight**: Provides statistical evidence regarding whether the latency distributions for different metrics resemble a normal distribution, which can be important for further statistical analysis assumptions.

### Two-Cluster Comparison Graphs

These graphs compare performance metrics between two different clusters (labeled as 'Cluster A' and 'Cluster B' in this analysis).

#### 1. Mean Bandwidth Comparison Between Clusters

* **File Name**: `cluster_comparison_mean_bw.png`
* **Description**: A side-by-side bar chart comparing the mean effective bandwidth (MB/s) for each matrix between the two clusters.
* **Insight**: Directly compares the average data transfer rates achieved by similar communication patterns on different cluster environments.

#### 2. Bandwidth Variance Comparison Between Clusters

* **File Name**: `cluster_comparison_variance_bw.png`
* **Description**: A side-by-side bar chart comparing the variance of the effective bandwidth for each matrix between the two clusters.
* **Insight**: Illustrates the consistency of bandwidth performance across iterations within each matrix for both clusters. Higher variance indicates more fluctuation.

#### 3. Coefficient of Variance of Effective BW Comparison Between Clusters

* **File Name**: `cluster_comparison_cov_bw.png`
* **Description**: A side-by-side bar chart comparing the Coefficient of Variance (CoV = Standard Deviation / Mean) of the effective bandwidth for each matrix between the two clusters.
* **Insight**: Provides a normalized measure of dispersion, allowing for a better comparison of relative variability in bandwidth performance between clusters, especially when their means differ significantly.

#### 4. Aggregated Latency Distribution Comparisons

* **File Name Pattern**: `cluster_comparison_{metric}_dists.png` (e.g., `cluster_comparison_max_latency_dists.png`)
* **Description**: These plots use Kernel Density Estimation (KDE) to compare the distributions of aggregated latency metrics (Max Latency, Min Latency, Median Latency) between the two clusters for each matrix. Solid lines represent Cluster A, and dashed lines represent Cluster B.
* **Insight**: Offers a visual comparison of the entire distribution shapes, revealing differences in skewness, modality, and spread of aggregated latencies between the two cluster environments.

#### 5. Combined Normality Test Results

* **File Name**: `normality_test_results_both_clusters.csv`
* **Description**: This CSV consolidates the normality test results (KS and SW statistics and p-values) for aggregated latency distributions from both clusters, including a 'Cluster' column to differentiate between Custer A and Cluster B.
* **Insight**: Allows for a direct statistical comparison of the normality of latency distributions between the two environments.
