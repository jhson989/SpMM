#include <iostream>
#include <vector>
#include <algorithm>
#include <CL/sycl.hpp>
namespace sycl=cl::sycl;

/*** Measure performance ***/
#include <sys/time.h>
#define ELAPSED_TIME(st, ed) ((ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6))
timeval start, end;

/*** Data configuration ***/
#define DTYPE int
const int M=1024, N=1024, K=1024;

/*** Debugging info ***/
//#define __MODE_DEBUG_TIME__
const int NUM_TESTS=2;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&,const std::vector<DTYPE>&);



int main (void) {


    std::cout << "=================================================\n";
    std::cout << "Parallel Sparse Matrix Multiplication via SYCL\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- Matrix : A["<<M<<","<<K<<"] * B["<<K<<","<<N<<"] = C["<<M<<","<<N<<"]\n";
    std::cout << "-- total size of three 2D matrices: "<<sizeof(DTYPE)*(M*N+M*K+K*N)/1024.0/1024.0/1024.0<<" GB\n";
    std::cout << "=================================================\n\n";

    /********************************************************
     *  SYCL setup
     ********************************************************/
    sycl::gpu_selector device;
    sycl::queue queue(device);

    /********************************************************
     *  Data initilzation
     ********************************************************/
    // Input data A
    std::vector<DTYPE> A(M*K);
    std::generate(A.begin(), A.end(), [](){return (std::rand()%100-50);});
    DTYPE* device_A = sycl::malloc_device<DTYPE>(M*K, queue);
    queue.memcpy(device_A, A.data(), M*K*sizeof(DTYPE));

    // Input data B
    std::vector<DTYPE> B(K*N);
    std::generate(B.begin(), B.end(), [](){return (std::rand()%100-50);});
    DTYPE* device_B = sycl::malloc_device<DTYPE>(K*N, queue);
    queue.memcpy(device_B, B.data(), K*N*sizeof(DTYPE));
    
    // Output data C
    std::vector<DTYPE> C(M*N);
    DTYPE* device_C = sycl::malloc_device<DTYPE>(M*N, queue);

    // For initial warming up

}
