struct Dim3 {
    uniform int x, y, z;
};
export void saxpy(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                  uniform float *x, uniform float *y, uniform int n,
                  uniform float a) {
    uniform int<3> blockIdx, threadIdx;

    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                         threadIdx.y++) {
                        for (threadIdx.x = 0; threadIdx.x < blockDim.x;
                             threadIdx.x++) {
                            int i = blockIdx.x * blockDim.x + threadIdx.x;
                            if (i < n)
                                y[i] = a * x[i] + y[i];
                        }
                    }
                }
            }
        }
    }
}
