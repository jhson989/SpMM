CC = nvcc
PROJECT = spmm
PROGRAM = ${PROJECT}.out
SRCS = ${PROJECT}.cu 
INCS = 

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${SRCS} ${INC} Makefile
	${CC} -o $@ ${SRCS}


run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}
