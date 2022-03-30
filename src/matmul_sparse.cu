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

    /*** Start of conversion ***/
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

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

    /*** End of conversion ***/
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" -- Elapsed time: %.3f s\n", msec_total*1e-3);


    #ifdef DEBUG_ON
    check_result(A, B, C);
    #endif

}
