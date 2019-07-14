CXXFLAGS=-std=c++17
SPMDFY="../../../build/spmdfy"
CUDA_PATH=/usr/local/cuda-9.0/
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
CUDA_LINK_FLAGS=-L ${CUDA_PATH}/lib64  -lcudart -I ../utils/


shmem: main.cu shared_memory.o shared_memory_ispc.o shared_memory.ispc
	clang++-9 ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -I "../../utils" shared_memory_ispc.o shared_memory.o main.cu -o shmem
shared_memory.o: shared_memory.cu
	clang++-9 ${CUDA_ARCH_FLAGS} ${CXXFLAGS} -o shared_memory.o -c shared_memory.cu
shared_memory_ispc.o: shared_memory.ispc
	ispc -h shared_memory_ispc.h -o shared_memory_ispc.o --target=avx2-i32x8 shared_memory.ispc
shared_memory.ispc: shared_memory.cu
	${SPMDFY} -o shared_memory.ispc -dump-json shared_memory.cu
clean:
	rm shared_memory.ispc shmem *.o *_ispc.h
