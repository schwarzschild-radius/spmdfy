CXXFLAGS=-std=c++17 -I ../utils/
CUDACC=clang++-9
CUDA_ARCH_FLAGS=
CUDA_LINK_FLAGS=
CUDA_PATH=/usr/local/cuda-9.0/
SPMDFY="../../build/spmdfy"
ifeq ($(CUDACC), clang++-9)
	CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
	CUDA_LINK_FLAGS=-L ${CUDA_PATH}lib64  -lcudart
endif

transpose: transpose_ispc.o transpose_cuda.o
	${CUDACC} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -I ../utils transpose_ispc.o transpose_cuda.o main.cu -o transpose
transpose_cuda.o: transpose.cu
	clang++-9 ${CUDA_ARCH_FLAGS} ${CXXFLAGS} -o transpose_cuda.o -c transpose.cu
transpose_ispc.o: transpose.ispc
	ispc --target avx2-i32x8 -h transpose.h -o transpose_ispc.o transpose.ispc
transpose.ispc: transpose.cu
	${SPMDFY} -o transpose.ispc transpose.cu -dump-json
clean:
	rm transpose transpose.ispc transpose.h transpose_ispc.o transpose_cuda.o