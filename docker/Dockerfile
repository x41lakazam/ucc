#FROM gitlab-master.nvidia.com:5005/dc_scale/cloudai_nccl_ucc:eyal-bisection-1
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Add UCC with transfer matrix in alltoallv perftest
RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        automake \
        autoconf \
        git \
        libtool \
        flex && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt && \
    git clone -b alltoallv_transfer_matrix_multi https://github.com/x41lakazam/ucc.git && cd ucc && \
    ./autogen.sh && \
    mkdir build && cd build && \
    ../configure --without-sharp \
                 --with-ucx=/opt/hpcx/ucx \
                 --with-cuda=/usr/local/cuda \
                 --with-mpi \
                 --with-nvcc-gencode="-gencode=arch=compute_90,code=sm_90" \
                 --prefix=/opt/ucc/build/install && \
    make -j install

RUN apt-get remove -y \
        autotools-dev \
        automake \
        autoconf \
        git \
        libtool \
        flex

RUN mkdir -p /var/lib/ucc_transfer_matrices
