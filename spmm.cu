#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <time.h>
// Debug
#define DEBUG_ON
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true);

/*******************************************************************
  * Matrix configuration
  ******************************************************************/

#define DTYPE float
const int M = 1024*1+1;
const int N = 1024*1+2;
const int K = 1024*1+3;
const float sparsity = 0.01;
const int warp_size = 32;




/*******************************************************************
  * Kernel code
  ******************************************************************/
template <typename T>
__global__ void get_num_nonzero_by_row(T* A, int* row_ptr, int M, int K) {


    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = threadIdx.x; // warp

    int K_under = (K/blockDim.x)*blockDim.x;
    int sum = 0;
    unsigned int bits;

    for (int i=0; i<K_under; i+=blockDim.x) {
        // warp communication
        bits = __ballot_sync(0xFFFFFFFF, A[y*K+(x+i)] != 0);
        if (x == 0){
            sum += __popc(bits);
        }
    }

    if (x == 0) {
        for (int i=K_under; i<K; i++) {
            if (A[y*K+i] != 0)
                sum++;
        }
        row_ptr[y+1] = sum;
    }
    
}


template <typename T, int WARP_SIZE>
__global__ void store_nonzero_by_row(T* A, int* row_ptr, int* col, T* value, int M, int K) {


    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = threadIdx.x; // warp

    __shared__ float svalue[WARP_SIZE];

    int K_under = (K/blockDim.x)*blockDim.x;
    int num = 0;

    for (int i=0; i<K_under; i+=blockDim.x) {
        // warp communication
        svalue[x] = A[y*K+(x+i)];
        __syncthreads();
        if (x == 0){
            
            #pragma unroll
            for (int w=0; w<WARP_SIZE; w++) {
                if (svalue[w] != 0) {
                    col[row_ptr[y]+num] = (w+i);
                    value[row_ptr[y]+num] = svalue[w];
                    num++;
                }
                    
            }

        }
    }

    if (x == 0) {
        for (int w=K_under; w<K; w++) {
            if (A[y*K+w] != 0){
                col[row_ptr[y]+num] = w;
                value[row_ptr[y]+num] = A[y*K+w];
                num++;   
            }
        }
    }


}

/*******************************************************************
  * Host code
  ******************************************************************/

DTYPE get_random_number() {return std::rand()%11-5;}
void make_sparse_matrix(std::vector<DTYPE>& A);
void convert_to_CSR(DTYPE* d_A, void** d_row_ptr_p, void** d_col_p, void** d_value_p);
void spmm_cpu(int* d_row_ptr, int* d_col, DTYPE* d_value, std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);

// for debug
void print_matrix(const std::vector<DTYPE>& A, int ROW);
template <typename T>
void print_vector(const std::vector<T>& A);
void check_csr(std::vector<DTYPE>& A, void** row_ptr_p, void** col_p, void** value_p, int ROW, int COL) ;
void check_result(std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);


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
    std::cout << "  -- Sparsity of matrix : " << sparsity << std::endl;
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
inline DTYPE get_nonzero() {
    if (std::rand()%2 == 0)
        return std::rand()%4+1;
    else
        return -1*(std::rand()%4+1);
}

void make_sparse_matrix(std::vector<DTYPE>& A) {

    int total_nonzero = sparsity * M * K;

    std::vector<int> idx(M*K);
    for (int i=0; i<M*K; i++) idx[i] = i;

    // Select matrix indices for nonzero value
    for (int j=0; j<20; j++)
        for (int i=0; i<total_nonzero; i++) {
            int random_idx = std::rand()%idx.size();
            DTYPE temp = idx[i];
            idx[i] = idx[random_idx];
            idx[random_idx] = temp;
        }

    // Fill nonzero value into selected indices
    for (int i=0; i<total_nonzero; i++) {
        A[idx[i]] = get_nonzero();
    }

}

void convert_to_CSR(DTYPE* d_A, void** d_row_ptr_p, void** d_col_p, void** d_value_p) {

    printf("CSR conversion launched...\n");

    
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    /*** Start of conversion ***/
    std::vector<int> row_ptr(M+1);
    cudaErrChk( cudaMalloc(d_row_ptr_p, sizeof(int)*(M+1)) );
    const dim3 dim_threads(warp_size, 1);
    const dim3 dim_blocks(1, M);
    
    // Count the number of non-zero values by rows
    get_num_nonzero_by_row<DTYPE><<<dim_blocks, dim_threads>>>(d_A, (int*)(*d_row_ptr_p), M, K);
    cudaErrChk( cudaMemcpy(row_ptr.data(),(*d_row_ptr_p), sizeof(int)*(M+1), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    // Exclusive scan
    row_ptr[0] = 0;
    for (int i=1; i<(M+1); i++)
        row_ptr[i] += row_ptr[i-1];

    cudaErrChk( cudaMemcpy((*d_row_ptr_p), row_ptr.data(), sizeof(int)*(M+1), cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    // Store non-zero values
    cudaErrChk( cudaMalloc(d_col_p, sizeof(int)*row_ptr[M]) );
    cudaErrChk( cudaMalloc(d_value_p, sizeof(DTYPE)*row_ptr[M]) );
    store_nonzero_by_row<DTYPE, warp_size><<<dim_blocks, dim_threads>>>(d_A, (int*)(*d_row_ptr_p), (int*)(*d_col_p), (DTYPE*)(*d_value_p), M, K);

    /*** End of conversion ***/
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" -- Elapsed time: %.3f s\n", msec_total*1e-3);

}



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






/*******************************************************************
  * Debug code
  ******************************************************************/

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) {
       fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

void print_matrix(const std::vector<DTYPE>& A, int ROW) {

    int COL = A.size() / ROW;

    for (int row=0; row<ROW; row++) {
        for (int col=0; col<COL; col++) {
            std::cout << A[row*COL+col] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void print_vector(const std::vector<T>& A) {

    for (int i=0; i<A.size(); i++)
        std::cout << A[i] << " ";

    std::cout << std::endl << std::endl;
}

void check_csr(std::vector<DTYPE>& A, void** row_ptr_p, void** col_p, void** value_p, int ROW, int COL) {


    std::vector<int> row_ptr(ROW+1);
    cudaErrChk( cudaMemcpy(row_ptr.data(),(*row_ptr_p), sizeof(int)*(M+1), cudaMemcpyDeviceToHost) );

    std::vector<int> col(row_ptr[ROW]);
    cudaErrChk( cudaMemcpy(col.data(),(*col_p), sizeof(int)*(row_ptr[ROW]), cudaMemcpyDeviceToHost) );


    std::vector<DTYPE> value(row_ptr[ROW]);
    cudaErrChk( cudaMemcpy(value.data(),(*value_p), sizeof(DTYPE)*(row_ptr[ROW]), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );


    
    print_matrix(A, ROW);
    print_vector(row_ptr);
    print_vector(col);
    print_vector(value);
    

    for (int r=0; r<ROW; r++) {
        for (int c=row_ptr[r]; c<row_ptr[r+1]; c++) {
            if (A[r*COL+col[c]] != value[c]){
                std::cout << " -- [[[ERR]]] Error occurs!!" << std::endl;
                return;
            }
            A[r*COL+col[c]] = 0;
        }
    }

    for (int i=0; i<A.size(); i++) {
        if (A[i]!=0) {
            std::cout << " -- [[[ERR]]] Error occurs!!" << std::endl;
            return;
        }            
    }
    std::cout << " -- No error!!" << std::endl;
}

void check_result(std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C) {

    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            DTYPE sum = 0;
            for (int k=0; k<K; k++)
                sum += (A[y*K+k]*B[k*N+x]);

            if (C[y*N+x] != sum) {
                std::cout << " -- [[[ERR]]] Error occurs!! : C["<<y<<","<<x<<"] = "<<C[y*N+x]<<" != gt("<<sum<<")" << std::endl;
                return;      
            }
        }
    }
    std::cout << " -- No error!!" << std::endl;
}