#!/bin/bash
#SBATCH --job-name=a2av_16N
#SBATCH --output=/mtrsysgwork/yyacobovich/install/all2allv/output_logs/12_06_25/uccalltoallv_16nodes_a2av_%j.txt
#SBATCH --error=/mtrsysgwork/yyacobovich/install/all2allv/output_logs/12_06_25/uccalltoallv_16nodes_a2av_%j.txt
#SBATCH --partition=ISR1-ALL
#SBATCH --reservation=sharonda_817
#SBATCH -N 16
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

CONTAINER="/mswg2/E2E/Regression_logs/squash/all2allv_ucc_perf_pytorch25_03_v1.4_inner_logs.sqsh"
BASEDIR="/path/to/all2allv"

# Note: the value for pairwise posts ($pw) and ib paths number ($qp) are
# what we think would work the best but is not tested yet.
############################ User configuration ###########################
# Path to the log reporting for each matrix independently
inner_log_dir="$BASEDIR/inner_logs"
# Path to the dir containing the transfer matrices
transfer_matrices_dir="$BASEDIR/transfer_matrices"
# Number of matrices
transfer_matrices_count=1
# Number of pairwise posts
pw=16
# Number of IB paths
qp=4
# Cluster partition
partition="batch_block1"
# Number of nodes to run the collective on
num_nodes=16
#######################################################################
echo "Node list: $SLURM_NODELIST"


# Loop through benchmarks and iterations
for bench in {0..3}; do
    for iter in {0..2}; do
        matrix_dir="bench${bench}_iter${iter}"
        echo "Running test for matrix directory: $matrix_dir"
        
        # Set correct transfer matrices directory based on benchmark number
        if [ $bench -eq 0 ]; then
            transfer_matrices_dir="$BASEDIR/transfer_matrices/$matrix_dir"
            log_suffix=""
        else
            transfer_matrices_dir="$BASEDIR/transfer_matrices/12_06_25/16N/$matrix_dir"
            log_suffix="_16N"
        fi

        UCC_PT_COLL_INNER_LOG_FILE=/workspace/inner_logs/matrices_exec_time${log_suffix}_${matrix_dir}.log \
        srun -p ISR1-ALL -N $num_nodes --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
            --container-mounts=$transfer_matrices_dir:/workspace/ucc_transfer_matrices,$inner_log_dir:/workspace/inner_logs \
            --container-image=$CONTAINER \
            bash -c '
            export UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT=1 \
            UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR="/workspace/ucc_transfer_matrices" \
            UCC_TL_UCP_ALLTOALLV_PAIRWISE_NUM_POSTS=0 \
            UCC_TL_UCP_TUNE=alltoallv:@pairwise \
            LD_LIBRARY_PATH=/workspace/ucc_files/lib:$LD_LIBRARY_PATH \
            UCX_TLS=rc,cuda_copy \
            UCX_RNDV_SCHEME=put_zcopy \
            UCX_RNDV_THRESH=0 \
            UCX_IB_GID_INDEX=3 \
            UCX_IB_TRAFFIC_CLASS="41" \
            CUDA_VISIBLE_DEVICES=0 \
            UCX_NET_DEVICES=mlx5_0:1 \
            UCC_TLS=ucp \
            UCX_IB_SL=0 \
            UCC_CL_BASIC_TLS=ucp
            /workspace/ucc_files/bin/ucc_perftest -c alltoallv -j $UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT -m cuda'
        
        wait
        echo "Completed test for matrix directory: $matrix_dir"
    done
done

echo "All tests completed"