#include <cuda_runtime.h>

struct A{
    __device__ __host__ A(int a){
        this->a = a;
    }
    int a;
    int b;
    int *c;
};