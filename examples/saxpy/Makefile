CXXFLAGS=-std=c++17 -I ../utils/
CUDACC=clang++-9
CUDA_ARCH_FLAGS=
CUDA_LINK_FLAGS=
CUDA_PATH=/usr/local/cuda-9.0/
SPMDFY="../../build/spmdfy"
ifeq ($(CUDACC), clang++-9)
	CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
	CUDA_LINK_FLAGS=-L ${CUDA_PATH}lib64  -lcudart
else
	CUDA_ARCH_FLAGS=-arch=sm_50
	CXXFLAGS=-std=c++14 -I ../utils
endif

saxpy: main.cu saxpy_ispc.o saxpy_cuda.o
	${CUDACC} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -I ../utils saxpy_ispc.o saxpy_cuda.o main.cu -o saxpy
saxpy_cuda.o: saxpy.cu
	clang++-9 ${CUDA_ARCH_FLAGS} ${CXXFLAGS} -o saxpy_cuda.o -c saxpy.cu
saxpy_ispc.o: saxpy.ispc
	ispc --target avx2-i32x8 -h saxpy.h -o saxpy_ispc.o saxpy.ispc
saxpy.ispc: saxpy.cu
	${SPMDFY} -o saxpy.ispc -dump-json saxpy.cu
clean:
	rm saxpy saxpy.ispc saxpy.h saxpy_ispc.o saxpy_cuda.o