struct Dim3 {
    uniform int x, y, z;
};
export void
transpose_parallel_per_element(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                               uniform unsigned int32 shared_memory_size,
                               uniform int a[], uniform int b[],
                               uniform size_t N, uniform size_t K) {
    unsigned int<3> blockIdx, threadIdx;

    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                         threadIdx.y++) {
                        for (threadIdx.x = programIndex;
                             threadIdx.x < blockDim.x;
                             threadIdx.x += programCount) {
                            int i = blockIdx.x * K + threadIdx.x;
                            int j = blockIdx.y * K + threadIdx.y;
                            b[j + i * N] = a[i + j * N];
                        }
                    }
                }
            }
        }
    }
}
