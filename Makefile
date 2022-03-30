CC = nvcc
PROJECT = spmm
PROGRAM = ${PROJECT}.out
MAIN = ${PROJECT}.cu 
SRCS = src/debug.cu src/convert.cu src/data.cu
INCS = include/config.cuh include/debug.cu include/convert.cuh include/data.cuh
DEBUG=OFF

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${MAIN} ${SRCS} ${INC} Makefile
	${CC} -o $@ ${MAIN} ${SRCS} -DDEBUG_${DEBUG}



run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}
