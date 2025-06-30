#!/bin/bash
#SBATCH --job-name=a2av_16N
#SBATCH --output=/mtrsysgwork/yyacobovich/install/all2allv/output_logs/12_06_25/uccalltoallv_16nodes_a2av_%j.txt
#SBATCH --error=/mtrsysgwork/yyacobovich/install/all2allv/output_logs/12_06_25/uccalltoallv_16nodes_a2av_%j.txt
#SBATCH --partition=ISR1-ALL
#SBATCH --reservation=sharonda_801
#SBATCH -N 16
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

CONTAINER="/mswg2/E2E/Regression_logs/squash/all2allv_ucc_perf_pytorch25_03_v1.4_inner_logs.sqsh"
BASEDIR="/mtrsysgwork/yyacobovich/install/all2allv"

# Note: the value for pairwise posts ($pw) and ib paths number ($qp) are
# what we think would work the best but is not tested yet.
############################ User configuration ###########################
# Path to the log reporting for each matrix independently
inner_log_dir="$BASEDIR/inner_logs/12_06_25/new_cc"
# Path to the dir containing the transfer matrices
transfer_matrices_dir="$BASEDIR/transfer_matrices/12_06_25/16N"
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


# Loop through matrix directories
for matrix_dir in bench0_iter{0,1,2}; do
    echo "Running test for matrix directory: $matrix_dir"
    transfer_matrices_dir="$BASEDIR/transfer_matrices/12_06_25/16N/$matrix_dir"
    UCC_PT_COLL_INNER_LOG_FILE=/workspace/inner_logs/matrices_exec_time_${matrix_dir}.log \
    srun -p ISR1-ALL -N $num_nodes --reservation=sharonda_801 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
        --container-mounts=$transfer_matrices_dir:/workspace/ucc_transfer_matrices,$inner_log_dir:/workspace/inner_logs \
        --container-image=$CONTAINER \
        bash -c '
        export UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT=1 \
        UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR="/workspace/ucc_transfer_matrices" \
        UCC_TL_UCP_ALLTOALLV_PAIRWISE_NUM_POSTS=0 \
        UCC_TL_UCP_TUNE=alltoallv:@pairwise \
        xUCC_COLL_TRACE=info \
        LD_LIBRARY_PATH=/workspace/ucc_files/lib:$LD_LIBRARY_PATH \
        UCX_TLS=rc,cuda_copy \
        UCX_RNDV_SCHEME=put_zcopy \
        UCX_IB_GID_INDEX=3 \
        UCX_IB_TRAFFIC_CLASS="41" \
        UCX_RNDV_THRESH=0 \
        xUCX_IB_ROCE_LOCAL_SUBNET=y \
        CUDA_VISIBLE_DEVICES=0 \
        UCX_NET_DEVICES=mlx5_0:1 \
        UCC_TLS=ucp \
        xUCC_LOG_LEVEL=info \
        UCX_IB_SL=0 \
        xUCX_LOG_LEVEL=debug \
        xUCX_LOG_FILE=/workspace/output_logs/ucx_log_%p_%h_file.txt \
        UCC_CL_BASIC_TLS=ucp \
        xUCC_CONFIG_FILE= \
        xUCX_LOG_LEVEL_TRIGGER=error \
        xUCX_HANDLE_ERRORS=freeze \
        xUCC_TL_NCCL_TUNE="0" 
        /workspace/ucc_files/bin/ucc_perftest -c alltoallv -j $UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT -m cuda'
    
    wait
    echo "Completed test for matrix directory: $matrix_dir"
done

for matrix_dir in bench1_iter{0,1,2}; do
    echo "Running test for matrix directory: $matrix_dir"
    transfer_matrices_dir="$BASEDIR/transfer_matrices/12_06_25/16N/$matrix_dir"
    UCC_PT_COLL_INNER_LOG_FILE=/workspace/inner_logs/matrices_exec_time_16N_${matrix_dir}.log \
    srun -p ISR1-ALL -N $num_nodes --reservation=sharonda_801 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
        --container-mounts=$transfer_matrices_dir:/workspace/ucc_transfer_matrices,$inner_log_dir:/workspace/inner_logs \
        --container-image=$CONTAINER \
        bash -c '
        export UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT=1 \
        UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR="/workspace/ucc_transfer_matrices" \
        UCC_TL_UCP_ALLTOALLV_PAIRWISE_NUM_POSTS=0 \
        UCC_TL_UCP_TUNE=alltoallv:@pairwise \
        xUCC_COLL_TRACE=info \
        LD_LIBRARY_PATH=/workspace/ucc_files/lib:$LD_LIBRARY_PATH \
        UCX_TLS=rc,cuda_copy \
        UCX_RNDV_SCHEME=put_zcopy \
        UCX_IB_GID_INDEX=3 \
        UCX_IB_TRAFFIC_CLASS="41" \
        UCX_RNDV_THRESH=0 \
        xUCX_IB_ROCE_LOCAL_SUBNET=y \
        CUDA_VISIBLE_DEVICES=0 \
        UCX_NET_DEVICES=mlx5_0:1 \
        UCC_TLS=ucp \
        xUCC_LOG_LEVEL=info \
        UCX_IB_SL=0 \
        xUCX_LOG_LEVEL=debug \
        xUCX_LOG_FILE=/workspace/output_logs/ucx_log_%p_%h_file.txt \
        UCC_CL_BASIC_TLS=ucp \
        xUCC_CONFIG_FILE= \
        xUCX_LOG_LEVEL_TRIGGER=error \
        xUCX_HANDLE_ERRORS=freeze \
        xUCC_TL_NCCL_TUNE="0" 
        /workspace/ucc_files/bin/ucc_perftest -c alltoallv -j $UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT -m cuda'
    
    wait
    echo "Completed test for matrix directory: $matrix_dir"
done

for matrix_dir in bench2_iter{0,1,2}; do
    echo "Running test for matrix directory: $matrix_dir"
    transfer_matrices_dir="$BASEDIR/transfer_matrices/12_06_25/16N/$matrix_dir"
    UCC_PT_COLL_INNER_LOG_FILE=/workspace/inner_logs/matrices_exec_time_16N_${matrix_dir}.log \
    srun -p ISR1-ALL -N $num_nodes --reservation=sharonda_801 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
        --container-mounts=$transfer_matrices_dir:/workspace/ucc_transfer_matrices,$inner_log_dir:/workspace/inner_logs \
        --container-image=$CONTAINER \
        bash -c '
        export UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT=1 \
        UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR="/workspace/ucc_transfer_matrices" \
        UCC_TL_UCP_ALLTOALLV_PAIRWISE_NUM_POSTS=0 \
        UCC_TL_UCP_TUNE=alltoallv:@pairwise \
        xUCC_COLL_TRACE=info \
        LD_LIBRARY_PATH=/workspace/ucc_files/lib:$LD_LIBRARY_PATH \
        UCX_TLS=rc,cuda_copy \
        UCX_RNDV_SCHEME=put_zcopy \
        UCX_IB_GID_INDEX=3 \
        UCX_IB_TRAFFIC_CLASS="41" \
        UCX_RNDV_THRESH=0 \
        xUCX_IB_ROCE_LOCAL_SUBNET=y \
        CUDA_VISIBLE_DEVICES=0 \
        UCX_NET_DEVICES=mlx5_0:1 \
        UCC_TLS=ucp \
        xUCC_LOG_LEVEL=info \
        UCX_IB_SL=0 \
        xUCX_LOG_LEVEL=debug \
        xUCX_LOG_FILE=/workspace/output_logs/ucx_log_%p_%h_file.txt \
        UCC_CL_BASIC_TLS=ucp \
        xUCC_CONFIG_FILE= \
        xUCX_LOG_LEVEL_TRIGGER=error \
        xUCX_HANDLE_ERRORS=freeze \
        xUCC_TL_NCCL_TUNE="0" 
        /workspace/ucc_files/bin/ucc_perftest -c alltoallv -j $UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT -m cuda'
    
    wait
    echo "Completed test for matrix directory: $matrix_dir"
done

for matrix_dir in bench3_iter{0,1,2}; do
    echo "Running test for matrix directory: $matrix_dir"
    transfer_matrices_dir="$BASEDIR/transfer_matrices/12_06_25/16N/$matrix_dir"
    UCC_PT_COLL_INNER_LOG_FILE=/workspace/inner_logs/matrices_exec_time_16N_${matrix_dir}.log \
    srun -p ISR1-ALL -N $num_nodes --reservation=sharonda_801 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
        --container-mounts=$transfer_matrices_dir:/workspace/ucc_transfer_matrices,$inner_log_dir:/workspace/inner_logs \
        --container-image=$CONTAINER \
        bash -c '
        export UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT=1 \
        UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR="/workspace/ucc_transfer_matrices" \
        UCC_TL_UCP_ALLTOALLV_PAIRWISE_NUM_POSTS=0 \
        UCC_TL_UCP_TUNE=alltoallv:@pairwise \
        xUCC_COLL_TRACE=info \
        LD_LIBRARY_PATH=/workspace/ucc_files/lib:$LD_LIBRARY_PATH \
        UCX_TLS=rc,cuda_copy \
        UCX_RNDV_SCHEME=put_zcopy \
        UCX_IB_GID_INDEX=3 \
        UCX_IB_TRAFFIC_CLASS="41" \
        UCX_RNDV_THRESH=0 \
        xUCX_IB_ROCE_LOCAL_SUBNET=y \
        CUDA_VISIBLE_DEVICES=0 \
        UCX_NET_DEVICES=mlx5_0:1 \
        UCC_TLS=ucp \
        xUCC_LOG_LEVEL=info \
        UCX_IB_SL=0 \
        xUCX_LOG_LEVEL=debug \
        xUCX_LOG_FILE=/workspace/output_logs/ucx_log_%p_%h_file.txt \
        UCC_CL_BASIC_TLS=ucp \
        xUCC_CONFIG_FILE= \
        xUCX_LOG_LEVEL_TRIGGER=error \
        xUCX_HANDLE_ERRORS=freeze \
        xUCC_TL_NCCL_TUNE="0" 
        /workspace/ucc_files/bin/ucc_perftest -c alltoallv -j $UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT -m cuda'
    
    wait
    echo "Completed test for matrix directory: $matrix_dir"
done


echo "All tests completed"