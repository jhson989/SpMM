#include "../include/matmul_sparse.cuh"


void spmm_cpu(int* d_row_ptr, int* d_col, DTYPE* d_value, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C) {

    printf("SpMM CPU version launched...\n");

    std::vector<int> row_ptr(M+1);
    cudaErrChk( cudaMemcpy(row_ptr.data(),d_row_ptr, sizeof(int)*(M+1), cudaMemcpyDeviceToHost) );

    std::vector<int> col(row_ptr[M]);
    cudaErrChk( cudaMemcpy(col.data(),d_col, sizeof(int)*(row_ptr[M]), cudaMemcpyDeviceToHost) );

    std::vector<DTYPE> value(row_ptr[M]);
    cudaErrChk( cudaMemcpy(value.data(),d_value, sizeof(DTYPE)*(row_ptr[M]), cudaMemcpyDeviceToHost) );

    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    /*** Start of matmul ***/
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    #pragma omp parallel for num_threads(8)
    for (int y=0; y<M; y++) {
        
        for (int x=0; x<N; x++) {
            DTYPE sum = 0;
            for(int c=row_ptr[y]; c<row_ptr[y+1]; c++) {
                int k = col[c];
                DTYPE v = value[c];
                sum += v*B[k*N+x];
            }
            C[y*N+x] = sum;
        }
        
    }

    /*** End of matmul ***/
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" -- Elapsed time: %.3f s\n", msec_total*1e-3);


    #ifdef DEBUG_ON
    check_result(A, B, C);
    #endif

}

__global__ void sparse_matmul_1(int* row_ptr, int* col, DTYPE* value, DTYPE* B, DTYPE* C, const int M, const int N) {

    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column

    if (y<M && x<N) {

        DTYPE sum = 0;
        for(int c=row_ptr[y]; c<row_ptr[y+1]; c++) {
            int k = col[c];
            DTYPE v = value[c];
            sum += v*B[k*N+x];
        }
        C[y*N+x] = sum;

    }
    

}


void spmm_gpu_1(int* d_row_ptr, int* d_col, DTYPE* d_value, DTYPE* d_A, DTYPE* d_B, DTYPE* d_C, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C) {

    printf("SpMM GPU version-1 launched...\n");

    const dim3 dim_threads(WARP_SIZE, WARP_SIZE);
    const dim3 dim_blocks((N+WARP_SIZE-1)/WARP_SIZE, (N+WARP_SIZE-1)/WARP_SIZE);

    /*** Start of matmul ***/
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    // GPU kernel
    sparse_matmul_1<<<dim_blocks, dim_threads>>>(d_row_ptr, d_col, d_value, d_B, d_C, M, N);
    cudaErrChk( cudaMemcpy(C.data(), d_C, sizeof(DTYPE)*(M*N), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    /*** End of matmul ***/
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" -- Elapsed time: %.3f s\n", msec_total*1e-3);




    #ifdef DEBUG_ON
    check_result(A, B, C);
    #endif

}