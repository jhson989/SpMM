#pragma once


#include "config.cuh"
#include "debug.cuh"



void convert_to_CSR(DTYPE* d_A, void** d_row_ptr_p, void** d_col_p, void** d_value_p);