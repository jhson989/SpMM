#pragma once

#include <iostream>
#include <vector>
#include <time.h>

#include "config.cuh"

// CUDA error handle
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
void cudaAssert(cudaError_t code, const char *file, int line);

// Check results
void check_csr(std::vector<DTYPE>& A, void** row_ptr_p, void** col_p, void** value_p, int ROW, int COL);
void check_result(std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);
