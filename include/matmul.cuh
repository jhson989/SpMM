#pragma once

#include "config.cuh"
#include "debug.cuh"


void spmm_cpu(int* d_row_ptr, int* d_col, DTYPE* d_value, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);