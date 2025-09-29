/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

 #include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>
#include <string>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>
#include <cerrno>
 
 ucc_pt_coll_alltoallv::ucc_pt_coll_alltoallv(ucc_datatype_t dt,
                          ucc_memory_type mt, bool is_inplace,
                          bool is_persistent, int shuffle_cols, int n_iters,
                          ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
 {
     has_inplace_   = true;
     has_reduction_ = false;
     has_range_     = false;
     has_bw_        = true;
     root_shift_    = 0;
     shuffle_cols_  = shuffle_cols;
     n_iters_       = n_iters;
 
     coll_args.mask                = UCC_COLL_ARGS_FIELD_FLAGS;
     coll_args.coll_type           = UCC_COLL_TYPE_ALLTOALLV;
     coll_args.src.info_v.datatype = dt;
     coll_args.src.info_v.mem_type = mt;
     coll_args.dst.info_v.datatype = dt;
     coll_args.dst.info_v.mem_type = mt;
     coll_args.flags               = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                                     UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
     if (is_inplace) {
         coll_args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
     }
 
     if (is_persistent) {
         coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
     }
 
 }

 double ucc_pt_coll_alltoallv::get_largest_rank(ucc_coll_args_t &args, int grsize) {
    double            S    = 0;
    size_t src_size = 0, dst_size = 0;
    int current_rank = comm->get_rank();


    for (int i = 0; i < grsize; i++) {
        if (i == current_rank) {
            continue; // skip self
        }
        src_size += ucc_coll_args_get_count(&args, args.src.info_v.counts, i);
        dst_size += ucc_coll_args_get_count(&args, args.dst.info_v.counts, i);
    }
    src_size *= ucc_dt_size(args.src.info_v.datatype);
    dst_size *= ucc_dt_size(args.dst.info_v.datatype);
    S = src_size > dst_size ? src_size : dst_size;
    return S;
 }
 
 double ucc_pt_coll_alltoallv::get_bw(float time_ms, double largest_rank, int grsize,
     ucc_pt_test_args_t test_args)
 {
     ucc_coll_args_t &args = test_args.coll_args;
     //float            N    = grsize;
     double            S    = 0;
     size_t src_size = 0, dst_size = 0;
     int current_rank = comm->get_rank();

     // Check if this is shuffle mode bandwidth calculation
     if (shuffle_cols_) {
         S = largest_rank;
         return (S / (time_ms * n_iters_)) / 1000.0;
     }
     
     // Original bandwidth calculation for non-shuffle mode
     for (int i = 0; i < grsize; i++) {
         if (i == current_rank) {
            std::cout << "skipping self" << std::endl;
            continue; // skip self
         }
         src_size += ucc_coll_args_get_count(&args, args.src.info_v.counts, i);
         dst_size += ucc_coll_args_get_count(&args, args.dst.info_v.counts, i);
     }
     src_size *= ucc_dt_size(args.src.info_v.datatype);
     dst_size *= ucc_dt_size(args.dst.info_v.datatype);
     S = src_size > dst_size ? src_size : dst_size;

     std::cout << "S: " << S << std::endl;
     std::cout << "time_ms: " << time_ms << std::endl;

     //return (S / time_ms) * ((N - 1) / N) / 1000.0;
     return (S / time_ms) / 1000.0;
 }
 
 double parse_transfer_matrix_token(std::string token)
 {
     size_t size;
     double val;
     try { 
         val = std::stod(token, &size);
     }
     catch (...) {
         throw std::invalid_argument("Invalid element in transfer matrix: " + token);
     }
 
     if (size < token.size())
     {
         switch(token[size])
         {
             case 'K':
                 val *= 1e3;
                 break;
             case 'M':
                 val *= 1e6;
                 break;
             case 'G':
                 val *= 1e9;
                 break;
             default:
                 throw std::invalid_argument("Unknown suffix from transfer matrix: " + token[size]);
         }
     }
         return val;
 }
 
 /**
 * Fill a matrix based on the passed file
 * The file should contain square matrices of size <comm_size>.
 * The rows of the matrix should be lines and the elements should be separated by a single space.
 * The element (i,j) represents the number of bytes rank i will send to rank j. 
 * The notation support convenient unit notation for gigabytes and megabytes, e.g 3G or 10M.
 */
 
 std::vector<std::vector<double>> transpose_transfer_matrix(std::vector<std::vector<double>>& transfer_matrix)
 {
     int N = transfer_matrix.size();
     // Create a new matrix with the same size
     std::vector<std::vector<double>> transposed_matrix(N, std::vector<double>(N, 0));
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             transposed_matrix[i][j] = transfer_matrix[j][i];
         }
     }
     return transposed_matrix;
 }
 
 
 void shuffle_matrix(std::vector<std::vector<double>>& transfer_matrix, int iter){
    // gets transposed matrix
    // shuffle the columns using uniform seed so all ranks have the same shuffle
    // Prefer a user-provided fixed seed if available, else fall back to iter
    unsigned long base_seed = static_cast<unsigned long>(iter);
    if (const char* env_seed = std::getenv("UCC_PT_SHUFFLE_SEED")) {
        char* endptr = nullptr;
        errno = 0;
        unsigned long parsed = std::strtoul(env_seed, &endptr, 10);
        if (errno == 0 && endptr && *endptr == '\0') {
            base_seed = parsed + static_cast<unsigned long>(iter);
        }
    }
    std::default_random_engine rng(static_cast<unsigned int>(base_seed));
    std::shuffle(transfer_matrix.begin(), transfer_matrix.end(), rng);
    // transpose the matrix back and assign to the original matrix
    transfer_matrix = transpose_transfer_matrix(transfer_matrix);
}
 
 
 void fill_transfer_matrix(std::vector<std::vector<double>>& transfer_matrix, std::string filename)
 {
     std::ifstream f;
     std::string line, token;
     std::istringstream linestream;
     int col = 0;
     int N = transfer_matrix.size();
 
     f.open(filename.c_str());
     if (!f.is_open())
         throw std::invalid_argument("Couldn't open transfer matrix file: " + filename);
     
     for (int row=0; row < N; row++){
         if (!getline(f, line))
             throw std::invalid_argument("Transfer matrix is expected to have " + std::to_string(N) + " rows but only have " + std::to_string(row+1));
 
         if (row >= N)
             throw std::invalid_argument("Transfer matrix rows number exceed expected number of " + std::to_string(N));
 
         linestream.str(line);
         linestream.clear();
         col = 0;
         while (linestream >> token){
             if (col >= N)
                 throw std::invalid_argument("Transfer matrix columns of row " + std::to_string(row+1) + " exceed expected number of " + std::to_string(N));
 
             transfer_matrix[row][col] = parse_transfer_matrix_token(token);
             col++;
         }
 
         if (col != N)
             throw std::invalid_argument("Transfer matrix row " + std::to_string(row+1) + " doesn't contain " + std::to_string(N) + " elements as expected.");
     }
 }
 
 
 void fill_transfer_matrices(std::vector<std::vector<std::vector<double>>>& transfer_matrices, std::string transfer_matrices_dir, int shuffle_cols)
 {
     std::string fn;
     std::exception_ptr exc;
 
     if (transfer_matrices_dir.back() != '/')
         transfer_matrices_dir.push_back('/');
         
     fn = transfer_matrices_dir + std::to_string(0);
     try{
         fill_transfer_matrix(transfer_matrices[0], fn);
     }
     catch (const std::exception& e){
         std::cerr << "Exception when trying to fill matrix number " << std::to_string(0) << ": " << e.what() << std::endl;
         throw;
     }
     //std::cout << "number of matrices: " << transfer_matrices.size() << std::endl;
     if (shuffle_cols && transfer_matrices.size() > 1){ 
         //std::cout << "Shuffling columns" << std::endl;   
         transfer_matrices[1] = transpose_transfer_matrix(transfer_matrices[0]);
         for (int mat_ix=1; mat_ix < transfer_matrices.size(); mat_ix++){
             transfer_matrices[mat_ix] = transfer_matrices[1];
             shuffle_matrix(transfer_matrices[mat_ix], mat_ix);
         }
         //std::cout << "Shuffled columns - done!" << std::endl;
     }
 }
 
 void ucc_pt_coll_alltoallv::print_transfer_matrix(const std::vector<std::vector<double>>& matrix, const std::string& title) {
     if (!title.empty()) {
         std::cout << title << std::endl;
     }
     for (const auto& row : matrix) {
         std::cout << "[";
         for (size_t i = 0; i < row.size(); ++i) {
             std::cout << row[i];
             if (i < row.size() - 1) {
                 std::cout << ", ";
             }
         }
         std::cout << "]" << std::endl;
     }
     std::cout << std::endl;
 }
 
 
 
 int count_files_in_dir(std::string path){    
     int count = 0;
     DIR* dir = opendir(path.c_str());
     if (!dir)
         throw std::invalid_argument("Can't read directory: " + path);
 
     struct dirent* entry;
     while ((entry = readdir(dir)) != nullptr) {
         if (entry->d_type == DT_REG) {
             count += 1;
         }
     }
     closedir(dir);
     return count;
 }
 
 void ucc_pt_coll_alltoallv::pre_run(ucc_coll_args_t &args, int iter, int shuffle_cols) {
     int                                 comm_size = comm->get_size();
     int                                 comm_rank = comm->get_rank();
     size_t                              dt_size   = ucc_dt_size(coll_args.src.info_v.datatype);
     int                                 matrix_ix = shuffle_cols ? iter : 0;
     int                                 src_displacement = 0;
     int                                 dst_displacement = 0;
     int                                 send_count, recv_count;
 
     if (matrix_ix > transfer_matrices.size())
         throw std::invalid_argument("Inner iteration is " + std::to_string(matrix_ix) + " but no matrix is available at this index.");
 
 
     for (int i = 0; i < comm_size; i++) {
         send_count = std::floor(transfer_matrices[matrix_ix][comm_rank][i] / dt_size);
         recv_count = std::floor(transfer_matrices[matrix_ix][i][comm_rank] / dt_size);
 
         ((uint32_t*)args.src.info_v.counts)[i] = send_count;
         ((uint32_t*)args.src.info_v.displacements)[i] = src_displacement;
         ((uint32_t*)args.dst.info_v.counts)[i] = recv_count;
         ((uint32_t*)args.dst.info_v.displacements)[i] = dst_displacement;
 
         src_displacement += send_count;
         dst_displacement += recv_count;
     }
     //std::cout << "iteration: " << iter << "rank: " << comm_rank << std::endl;
     //std::cout << "matrix_ix: " << matrix_ix << std::endl;
     //std::cout << "shuffle_cols: " << shuffle_cols << std::endl;
     //print_transfer_matrix(transfer_matrices[matrix_ix], "Transfer matrix");
 }
 
 ucc_status_t ucc_pt_coll_alltoallv::init_args(size_t count, ucc_pt_test_args_t &test_args)
 {
     ucc_coll_args_t                     &args      = test_args.coll_args;
     int                                 comm_size = comm->get_size();
     int                                 comm_rank = comm->get_rank();
     size_t                              dst_header_size, src_header_size, max_dst_header_size, max_src_header_size;
     ucc_status_t                        st        = UCC_OK;
     int                                 n_matrices = 1;
     std::string                         transfer_matrices_dir;
 
    //  // Temporary: Forbid usage without transfer matrices
    //  if (!std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR")){
    //      std::cerr << "Required environment variable was not provided: UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR" << std::endl;
    //      std::terminate();
    //  }
    //  //End of temporary snippet 
 
    //  if (std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT")){
    //      if (!std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR"))
    //          throw std::invalid_argument("Environment variable UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT is set but UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR was not.");
 
    //      transfer_matrices_dir = std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR");
    //      n_matrices = atoi(std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT"));
    //      if (n_matrices == 0)
    //          throw std::invalid_argument("Invalid value for environment variable UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT");
         
    //      if (n_matrices != count_files_in_dir(transfer_matrices_dir))
    //          throw std::invalid_argument("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_COUNT doesn't match the count of files in UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR");
    //  }
 
     // if shuffle cols is set, we need to use the number of iterations as the number of matrices
     //std::cout << "shuffle_cols_: " << shuffle_cols_ << std::endl;
     //std::cout << "n_iters_: " << n_iters_ << std::endl;
     if (shuffle_cols_ && (n_iters_ > 1)){
         //std::cout << "resizing number of matrices to " << n_iters_ << std::endl;
         n_matrices = n_iters_;
     }
     transfer_matrices.resize(n_matrices, std::vector<std::vector<double>>(comm_size, std::vector<double>(comm_size, count)));
     //std::cout << "transfer_matrices.size(): " << transfer_matrices.size() << std::endl;
    if (std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR")) {
        transfer_matrices_dir = std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRICES_DIR");
        fill_transfer_matrices(transfer_matrices, transfer_matrices_dir, shuffle_cols_);
    }
     
     // Add a barrier so that all ranks complete init_args before any proceed
     comm->barrier();
     max_src_header_size = max_dst_header_size = 0;
     for (int mat_ix=0; mat_ix < transfer_matrices.size(); mat_ix++){
         src_header_size = dst_header_size = 0;
         for (size_t i=0; i < comm_size; i++)
             src_header_size += transfer_matrices[mat_ix][comm_rank][i];
         
         for (size_t i=0; i < comm_size; i++)
             dst_header_size += transfer_matrices[mat_ix][i][comm_rank];
         
         if (src_header_size > max_src_header_size)
             max_src_header_size = src_header_size;
 
         if (dst_header_size > max_dst_header_size)
             max_dst_header_size = dst_header_size;
     }
 
     args = coll_args;
     args.src.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
     UCC_MALLOC_CHECK_GOTO(args.src.info_v.counts, exit, st);
     args.src.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
     UCC_MALLOC_CHECK_GOTO(args.src.info_v.displacements, free_src_count, st);
     args.dst.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
     UCC_MALLOC_CHECK_GOTO(args.dst.info_v.counts, free_src_displ, st);
     args.dst.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
     UCC_MALLOC_CHECK_GOTO(args.dst.info_v.displacements, free_dst_count, st);
     UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, max_dst_header_size, args.dst.info_v.mem_type),
                   free_dst_displ, st);
     args.dst.info_v.buffer = dst_header->addr;
     if (!UCC_IS_INPLACE(args)) {
         UCCCHECK_GOTO(ucc_pt_alloc(&src_header, max_src_header_size, args.src.info_v.mem_type),
                       free_dst, st);
         args.src.info_v.buffer = src_header->addr;
     }
 
     return UCC_OK;
 free_dst:
     ucc_pt_free(dst_header);
 free_dst_displ:
     ucc_free(args.dst.info_v.displacements);
 free_dst_count:
     ucc_free(args.dst.info_v.counts);
 free_src_displ:
     ucc_free(args.src.info_v.displacements);
 free_src_count:
     ucc_free(args.src.info_v.counts);
 exit:
     return st;
 }
 
 void ucc_pt_coll_alltoallv::free_args(ucc_pt_test_args_t &test_args)
 {
     ucc_coll_args_t &args = test_args.coll_args;
 
     if (!UCC_IS_INPLACE(args)) {
         ucc_pt_free(src_header);
     }
     ucc_pt_free(dst_header);
     ucc_free(args.dst.info_v.counts);
     ucc_free(args.dst.info_v.displacements);
     ucc_free(args.src.info_v.counts);
     ucc_free(args.src.info_v.displacements);
 }