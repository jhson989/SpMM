#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>


#include "include/config.cuh"
#include "include/debug.cuh"
#include "include/convert.cuh"
#include "include/data.cuh"


/*******************************************************************
  * Host code
  ******************************************************************/
void spmm_cpu(int* d_row_ptr, int* d_col, DTYPE* d_value, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);



/*******************************************************************
  * Main
  ******************************************************************/

int main(void) {

    srand(time(NULL));
    std::cout << "" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Sparse Matrix Multipliation Example" << std::endl;
    std::cout << "  -- Mutiplication of a sparse matrix and a dense matrix" << std::endl;
    std::cout << "  -- C["<<M<<","<<N<<"] = A["<<M<<","<<K<<"] * B["<<K<<","<<N<<"]" << std::endl;
    std::cout << "  -- Sparsity of matrix : " << SPARSITY << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "" << std::endl;

    /* Data initialization */
    std::vector<DTYPE> A(M*K);
    make_sparse_matrix(A);
    std::vector<DTYPE> B(K*N);
    std::generate(B.begin(), B.end(), get_random_number);
    std::vector<DTYPE> C(M*N);

    /* Alloc GPU memory */
    DTYPE *d_A, *d_B, *d_C;
    cudaErrChk( cudaMalloc((void**)&d_A, sizeof(DTYPE)*M*K) );
    cudaErrChk( cudaMalloc((void**)&d_B, sizeof(DTYPE)*K*N) );
    cudaErrChk( cudaMalloc((void**)&d_C, sizeof(DTYPE)*M*N) );
    
    /* Memcpy from host to device */
    cudaErrChk( cudaMemcpy(d_A, A.data(), sizeof(DTYPE)*M*K, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaMemcpy(d_B, B.data(), sizeof(DTYPE)*K*N, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );


    /*****************************
     * Conversion 
     *****************************/
    int *d_row_ptr, *d_col;
    DTYPE *d_value;
    convert_to_CSR(d_A, (void**)&d_row_ptr, (void**)&d_col, (void**)&d_value);

    /*****************************
     * Sparse - Dense Matrix Multiplication
     *****************************/
    spmm_cpu(d_row_ptr, d_col, d_value, A, B, C);


    /* Finalize */
    cudaErrChk( cudaFree(d_A) );
    cudaErrChk( cudaFree(d_B) );
    cudaErrChk( cudaFree(d_C) );
    cudaErrChk( cudaFree(d_row_ptr) );
    cudaErrChk( cudaFree(d_col) );
    cudaErrChk( cudaFree(d_value) );
    return 0;
}


/*******************************************************************
  * Host code
  ******************************************************************/



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




