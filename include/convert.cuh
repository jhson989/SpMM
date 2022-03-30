#pragma once

#include "config.cuh"
#include "debug.cuh"

/********************************************************************************
  * CSR convertor funciton
  *******************************************************************************/

// Dense matrix A -> CSR format [rowPtr, col, and value] arrays
void convert_to_CSR(DTYPE* d_A, void** d_row_ptr_p, void** d_col_p, void** d_value_p);