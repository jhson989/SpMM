#pragma once

/********************************************************************************
  * Matrix configuration
  * C[M,N] = A[M,K]*B[K,N]
  *******************************************************************************/

#define DTYPE float // matrix data type
const int M = 10240; // height of C and A
const int N = 10240; // width of C and B
const int K = 10240; // width of A and height of B
const float SPARSITY = 0.01; // Sparsity : (the number of zero-valued elements) / (the total number of elements)
const int WARP_SIZE = 32; // GPU warp size. It depends on GPU architecture