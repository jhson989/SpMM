CC = nvcc
PROJECT = spmm
PROGRAM = ${PROJECT}.out
MAIN = ${PROJECT}.cu 
SRCS = src/debug.cu src/convert.cu src/data.cu src/matmul_sparse.cu src/matmul_dense.cu
INCS = include/config.cuh include/debug.cuh include/convert.cuh include/data.cuh include/matmul_sparse.cuh include/matmul_dense.cuh
DEBUG=OFF
OPTS = -Xcompiler -fopenmp

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${MAIN} ${SRCS} ${INCS} Makefile
	${CC} -o $@ ${MAIN} ${SRCS} -DDEBUG_${DEBUG} ${OPTS}



run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}
