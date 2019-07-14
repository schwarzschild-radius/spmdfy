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

finite_difference: finite_difference_ispc.o
	${CUDACC} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} finite_difference_ispc.o finite_difference.cu main.cu -o finite_difference
finite_difference_ispc.o: finite_difference.ispc
	ispc --target avx2-i32x8 -h finite_difference_ispc.h -o finite_difference_ispc.o finite_difference.ispc
finite_difference.ispc: finite_difference.cu
	${SPMDFY} -o finite_difference.ispc finite_difference.cu -dump-json
clean:
	rm *.o finite_difference finite_difference.ispc *_ispc.h