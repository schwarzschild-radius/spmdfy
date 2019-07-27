CXXFLAGS=-std=c++17
SPMDFY="../../../build/spmdfy"
CUDA_PATH=/usr/local/cuda-9.0/
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
CUDA_LINK_FLAGS=-L ${CUDA_PATH}/lib64  -lcudart -I ../utils/
CUDA_CLANG=clang++-9
ISPC=ispc

atomic: main.cu atomic_cuda.o atomic_ispc.o
	${CUDA_CLANG} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -I "../../utils" atomic_ispc.o atomic_cuda.o main.cu -o atomic
atomic_cuda.o: atomic.cu
	${CUDA_CLANG} ${CUDA_ARCH_FLAGS} ${CXXFLAGS} -o atomic_cuda.o -c atomic.cu
atomic_ispc.o: atomic.ispc
	${ISPC} -h atomic.h -o atomic_ispc.o --target=avx2-i32x8 atomic.ispc
atomic.ispc: atomic.cu
	${SPMDFY} -o atomic.ispc -dump-json atomic.cu
clean:
	rm atomic.ispc atomic atomic_cuda.o atomic_ispc.o atomic.h
