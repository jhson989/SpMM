/********************************************************************************
  * Main
  * Sparse Matrix Multipliation Example
  * Author : Janghyun Son
  * Email : jhson989@gmail.com
  *******************************************************************************/


#include <algorithm>
#include <time.h>

#include "include/config.cuh" // Program configuration 
#include "include/debug.cuh"  // Debug code
#include "include/data.cuh" // Sparse matrix generator
#include "include/convert.cuh" // CSR convertor
#include "include/matmul_sparse.cuh" // SpMM implementation



int main(void) {

    /*******************************************************************
     * Log 
     *******************************************************************/

    srand(time(NULL));
    std::cout << "" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Sparse Matrix Multipliation Example" << std::endl;
    std::cout << "  -- Mutiplication of a sparse matrix and a dense matrix" << std::endl;
    std::cout << "  -- C["<<M<<","<<N<<"] = A["<<M<<","<<K<<"] * B["<<K<<","<<N<<"]" << std::endl;
    std::cout << "  -- Sparsity of matrix : " << SPARSITY << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "" << std::endl;



    /*******************************************************************
     * Data initialization
     *******************************************************************/

    /* Host data generation */
    std::vector<DTYPE> A(M*K);
    make_sparse_matrix(A);
    std::vector<DTYPE> B(K*N);
    std::generate(B.begin(), B.end(), get_random_number);
    std::vector<DTYPE> C(M*N, 0);

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



    /*******************************************************************
     * Conversion 
     *******************************************************************/
    
    /* Device memory for CSR format array : rowPtr, col, value */
    int *d_row_ptr, *d_col; 
    DTYPE *d_value;
    
    /* Run CSR convertor */
    convert_to_CSR(d_A, (void**)&d_row_ptr, (void**)&d_col, (void**)&d_value);



    /*******************************************************************
     * Sparse - Dense Matrix Multiplication
     *******************************************************************/

    /* Run SpMM CPU implementation */
    spmm_cpu(d_row_ptr, d_col, d_value, A, B, C);

    /* Run SpMM GPU implementation - 1 */
    spmm_gpu_1(d_row_ptr, d_col, d_value, d_A, d_B, d_C, A, B, C);

    /*******************************************************************
     * Finalize
     *******************************************************************/

    /* Dealloc memory */
    cudaErrChk( cudaFree(d_A) );
    cudaErrChk( cudaFree(d_B) );
    cudaErrChk( cudaFree(d_C) );
    cudaErrChk( cudaFree(d_row_ptr) );
    cudaErrChk( cudaFree(d_col) );
    cudaErrChk( cudaFree(d_value) );
    return 0;
}
  