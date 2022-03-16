
  
# CMake bianry version
cmake_minimum_required(VERSION 3.20)
project(spmm)

set(PATH_SYCL_BUILD /data/share/oneapi/llvm/build)
set(CMAKE_CXX_COMPILER ${PATH_SYCL_BUILD}/bin/clang++)
set(PATH_SYCL_INC ${PATH_SYCL_BUILD}/include/sycl)
set(PATH_SYCL_LIB ${PATH_SYCL_BUILD}/lib)
set(SYCL_COMPILE_OPTION -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

set(APP ${CMAKE_PROJECT_NAME}.out)
set(MAIN ${CMAKE_PROJECT_NAME}.cpp)

add_executable(${APP} ${MAIN})
target_include_directories(${APP} PUBLIC ${PATH_SYCL_INC})
target_compile_options(${APP} PUBLIC ${SYCL_COMPILE_OPTION})
target_link_libraries(${APP} PUBLIC sycl)
target_link_directories(${APP} PUBLIC ${PATH_SYCL_LIB})
target_link_options(${APP} PUBLIC ${SYCL_COMPILE_OPTION})