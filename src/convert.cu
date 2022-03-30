#include "../include/convert.cuh"



/*******************************************************************
  * Kernel code
  ******************************************************************/

  
__global__ void get_num_nonzero_by_row(DTYPE* A, int* row_ptr, int M, int K) {


    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = threadIdx.x; // warp

    int K_under = (K/blockDim.x)*blockDim.x;
    int sum = 0;
    unsigned int bits;

    for (int i=0; i<K_under; i+=blockDim.x) {
        // warp communication
        bits = __ballot_sync(0xFFFFFFFF, A[y*K+(x+i)] != 0);
        if (x == 0){
            sum += __popc(bits);
        }
    }

    if (x == 0) {
        for (int i=K_under; i<K; i++) {
            if (A[y*K+i] != 0)
                sum++;
        }
        row_ptr[y+1] = sum;
    }
    
}


__global__ void store_nonzero_by_row(DTYPE* A, int* row_ptr, int* col, DTYPE* value, int M, int K) {


    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = threadIdx.x; // warp

    __shared__ float svalue[WARP_SIZE];

    int K_under = (K/blockDim.x)*blockDim.x;
    int num = 0;

    for (int i=0; i<K_under; i+=blockDim.x) {
        // warp communication
        svalue[x] = A[y*K+(x+i)];
        __syncthreads();
        if (x == 0){
            
            #pragma unroll
            for (int w=0; w<WARP_SIZE; w++) {
                if (svalue[w] != 0) {
                    col[row_ptr[y]+num] = (w+i);
                    value[row_ptr[y]+num] = svalue[w];
                    num++;
                }
                    
            }

        }
    }

    if (x == 0) {
        for (int w=K_under; w<K; w++) {
            if (A[y*K+w] != 0){
                col[row_ptr[y]+num] = w;
                value[row_ptr[y]+num] = A[y*K+w];
                num++;   
            }
        }
    }


}



void convert_to_CSR(DTYPE* d_A, void** d_row_ptr_p, void** d_col_p, void** d_value_p) {

    printf("CSR conversion launched...\n");

    
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    /*** Start of conversion ***/
    std::vector<int> row_ptr(M+1);
    cudaErrChk( cudaMalloc(d_row_ptr_p, sizeof(int)*(M+1)) );
    const dim3 dim_threads(WARP_SIZE, 1);
    const dim3 dim_blocks(1, M);
    
    // Count the number of non-zero values by rows
    get_num_nonzero_by_row<<<dim_blocks, dim_threads>>>(d_A, (int*)(*d_row_ptr_p), M, K);
    cudaErrChk( cudaMemcpy(row_ptr.data(),(*d_row_ptr_p), sizeof(int)*(M+1), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    // Exclusive scan
    row_ptr[0] = 0;
    for (int i=1; i<(M+1); i++)
        row_ptr[i] += row_ptr[i-1];

    cudaErrChk( cudaMemcpy((*d_row_ptr_p), row_ptr.data(), sizeof(int)*(M+1), cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    // Store non-zero values
    cudaErrChk( cudaMalloc(d_col_p, sizeof(int)*row_ptr[M]) );
    cudaErrChk( cudaMalloc(d_value_p, sizeof(DTYPE)*row_ptr[M]) );
    store_nonzero_by_row<<<dim_blocks, dim_threads>>>(d_A, (int*)(*d_row_ptr_p), (int*)(*d_col_p), (DTYPE*)(*d_value_p), M, K);

    /*** End of conversion ***/
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" -- Elapsed time: %.3f s\n", msec_total*1e-3);

}
