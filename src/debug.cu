#include "../include/debug.cuh"

/*******************************************************************
  * Debug code
  ******************************************************************/


// CUDA error handle
void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
       fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

// Check the csr conversion correctness
void check_csr(std::vector<DTYPE>& A, void** row_ptr_p, void** col_p, void** value_p, int ROW, int COL) {


    std::vector<int> row_ptr(ROW+1);
    cudaErrChk( cudaMemcpy(row_ptr.data(),(*row_ptr_p), sizeof(int)*(M+1), cudaMemcpyDeviceToHost) );

    std::vector<int> col(row_ptr[ROW]);
    cudaErrChk( cudaMemcpy(col.data(),(*col_p), sizeof(int)*(row_ptr[ROW]), cudaMemcpyDeviceToHost) );


    std::vector<DTYPE> value(row_ptr[ROW]);
    cudaErrChk( cudaMemcpy(value.data(),(*value_p), sizeof(DTYPE)*(row_ptr[ROW]), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    for (int r=0; r<ROW; r++) {
        for (int c=row_ptr[r]; c<row_ptr[r+1]; c++) {
            if (A[r*COL+col[c]] != value[c]){
                std::cout << " -- [[[ERR]]] Error occurs !!!" << std::endl;
                return;
            }
            A[r*COL+col[c]] = 0;
        }
    }

    for (int i=0; i<A.size(); i++) {
        if (A[i]!=0) {
            std::cout << " -- [[[ERR]]] Error occurs !!!" << std::endl;
            return;
        }            
    }
    std::cout << " -- Check correctness: no error !!!" << std::endl;
}


// Make sure the correctness of SpMM
void check_result(std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C) {

    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            DTYPE sum = 0;
            for (int k=0; k<K; k++)
                sum += (A[y*K+k]*B[k*N+x]);

            if (C[y*N+x] != sum) {
                std::cout << " -- [[[ERR]]] Error occurs !!! : C["<<y<<","<<x<<"] = "<<C[y*N+x]<<" != gt("<<sum<<")" << std::endl;
                return;      
            }
        }
    }
    std::cout << " -- Check correctness: no error !!!" << std::endl;
}