#pragma once

#include "config.cuh"
#include "debug.cuh"


void demm_gpu_1(DTYPE* d_A, DTYPE* d_B, DTYPE* d_C, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);