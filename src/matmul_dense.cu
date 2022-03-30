#include "../include/matmul_sparse.cuh"



__global__ void dense_matmul_1(DTYPE* A, DTYPE* B, DTYPE* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column

    if (y<M && x<N) {

        DTYPE sum = 0;
        for (int k=0; k<K; k++) {
            sum += A[y*K+k]*B[k*N+x];
        }
        C[y*N+x] = sum;
    }
    
}
void demm_gpu_1(DTYPE* d_A, DTYPE* d_B, DTYPE* d_C, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C) {
    
    printf("DeMM GPU version-1 launched...\n");

    const dim3 dim_threads(WARP_SIZE, WARP_SIZE);
    const dim3 dim_blocks((N+WARP_SIZE-1)/WARP_SIZE, (N+WARP_SIZE-1)/WARP_SIZE);

    /*** Start of matmul ***/
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    // GPU kernel
    dense_matmul_1<<<dim_blocks, dim_threads>>>(d_A, d_B, d_C, M, N, K);
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