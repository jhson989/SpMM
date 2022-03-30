#pragma once
#include <vector>
#include "config.cuh"

/********************************************************************************
  * Data initialization
  *******************************************************************************/

DTYPE get_random_number(); // return random number for random-value initialization
void make_sparse_matrix(std::vector<DTYPE>& A); // Generate unstructred sparse matrix in dense matrix format (2d array)