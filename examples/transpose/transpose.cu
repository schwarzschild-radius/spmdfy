__global__ void transpose_parallel_per_element(int a[], int b[], size_t N,
                                               size_t K) {
    int i = blockIdx.x * K + threadIdx.x;
    int j = blockIdx.y * K + threadIdx.y;

    b[j + i * N] = a[i + j * N];
}