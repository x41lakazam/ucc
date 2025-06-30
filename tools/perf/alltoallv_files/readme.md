## Implementation 
Currently ucc perftest runs N iterations for every message size, we decided to include all the alltoallv executions in each iteration. The run_single_coll_test method is responsible for running the N iterations, from now on let's denote those iterations the main iterations.
To do so we added an inner loop to run_single_coll_test, this loop is executing alltoallv with every transfer matrix provided. Therefore in each main iteration, all the matrices are executed.
At the beginning of the test, every matrix file is read and the matrices are stored in the memory. Then in the inner loop a method called coll->pre_run has been added to modify the collective arguments accordingly to the right matrix, this method is ran in each inner loop before the collective execution.
The send and receives buffers are allocated once in the beginning of the test, their allocation size is the biggest possible size (calculated across the matrices). Note that with a real workload, there might be additional time for the allocation & registration of these buffers (highly optimized in pytorch), this is not the case here.

## Output interpertation
Bottom line: the "max time" is the relevant metric, it represents the time it took for the whole collective to finish.
The main output (in stdout) is printing the average, min and max time it took the ranks to execute all the matrices, averaged over the main iterations (controlled by -n and -w). Therefore the only relevant piece here is the max time, as it represents the time it took, in average, to execute every collectives (i.e every matrix).
Note that after each alltoallv execution, a barrier is executed, therefore the ranks start at the same time and the maximum time represents the time since the first rank started the collective until the last rank finished. By the way, the time measurement don't include the barrier.

**Inner logs: in addition to the output log mentioned above, the test will give inner log files (in the path named in UCC_PT_COLL_INNER_LOG_FILE parameter). You will get a file for each rank (number of the ranl will be mentioned after the .log). In each file there will be a line for each iteration with that rank's latency for this iteration (in us).

## Parameters
The transfer matrix should be written in a file where each row contains the elements separated by a space, each element is a number of bytes. It support convenient unit usage of megabytes and gigabytes, in the format 1G or 5M.
There is support for multiple transfer matrices, which will be executed one after the other. All the transfer matrices should be in a directory and the path to this directory needs to be passed to the environment variable UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR. The directory should contain only the matrices files, and the number of matrices should be passed in the environment variable. UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT.
Each transfer matrix should be in a file named after the index of the matrix, therefore the name of the file should be a number between 0 and UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT-1 (the directory should contain files named 0, 1, ...).
Note that if UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT doesn't match the number of files in the directory, an error will be thrown.

The -b (min_count) and -b (max_count) arguments of ucc perfest are not relevant here and should be set to 0 (-b 0 -e 0). This is because they control the message size and here we use the matrices to control this.

The -j (n_inner_iter) argument should be the same as UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT.

The inner loop log absolute path should be provided in the environment variable UCC_PT_COLL_INNER_LOG_FILE.

## Matrix Generator
Jonathan Paul's tool for generating matrices: https://gitlab-master.nvidia.com/e2e-arch-network/a2aV_analysis_misc_tools

## Analyzing Scripts
Iv'e added 2 python scripts that will help with extracting data from the inner logs files and creating informative graphs.
1. ### Inner logs extraction from directory
[inner_logs_extract.py](https://github.com/x41lakazam/ucc/blob/yael_updates/tools/perf/alltoallv_files/inner_logs_extract.py) - input path should point to a folder with all the inner log files, output path will be a csv file that combines all files into 1 table.

2. ### Create analyzing graphs
[graph_from_inner_log.py](https://github.com/x41lakazam/ucc/blob/yael_updates/tools/perf/alltoallv_files/graph_from_inner_log.py) -
This script analyzes alltoallv performance data and generates a variety of informative graphs and CSVs. Below is a summary of the outputs:

#### Single-Cluster Analysis Graphs

- **Individual Latency Histograms**  
  Shows the distribution of individual latency measurements for each matrix.  
  *File: `matrix{id}_histogram.png`*

- **Latency Comparison Boxplot**  
  Compares latency distributions across all matrices in a single boxplot.  
  *File: `boxplot_latency_comparison.png`*

- **Aggregated Latency Histograms**  
  For each matrix, shows histograms of max, min, and median latency per iteration.  
  *Files: `matrix_{id}_{metric}_histogram.png` in `aggregated_latency_histograms/`*

- **Message Size vs Latency Dual-Axis Charts**  
  Dual-axis bar charts showing message size sent (MB) and average latency (μs) per rank.  
  *File: `matrix{id}_sent_vs_latency.png`*

- **Effective Bandwidth per Rank**  
  Bar chart of effective bandwidth (MB/s) for each rank.  
  *File: `matrix{id}_bandwidth.png`*

#### Two-Cluster Comparison Graphs

- **Mean Bandwidth Comparison**  
  Side-by-side bar chart comparing average effective bandwidth between clusters.  
  *File: `cluster_comparison_mean_bw.png`*

- **Bandwidth Variance Comparison**  
  Side-by-side bar chart comparing bandwidth variance between clusters.  
  *File: `cluster_comparison_variance_bw.png`*

- **Coefficient of Variation Comparison**  
  Side-by-side bar chart comparing normalized bandwidth variability (std/mean).  
  *File: `cluster_comparison_cov_bw.png`*

- **Distribution Comparison Plots**  
  KDE plots comparing the shapes of max, min, and median latency distributions between clusters.  
  *Files: `cluster_comparison_{metric}_dists.png`*

#### Additional Outputs

- **Job Completion Time CSV**  
  `jct_results.csv`: Average max latency per iteration (Job Completion Time) for each matrix.
- **Enriched Data CSV**  
  `enriched_average_latencies.csv`: Data with calculated bandwidth and message size metrics.
- **Normality Test Results**  
  `normality_test_results.csv` and `normality_test_results_both_clusters.csv`: Results of statistical normality tests on latency distributions.
