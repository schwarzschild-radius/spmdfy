struct Dim3 {
    uniform int x, y, z;
};
export void atomic_cuda(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                        uniform int d_bins[], uniform const int d_in[],
                        uniform const int BIN_COUNT) {
    uniform int<3> blockIdx, threadIdx;

    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                         threadIdx.y++) {
                        for (threadIdx.x = 0; threadIdx.x < blockDim.x;
                             threadIdx.x++) {
                            int myId = threadIdx.x + blockDim.x * blockIdx.x;
                            int myItem = d_in[myId];
                            int myBin = myItem % BIN_COUNT;
                            atomic_add_global(&(d_bins[myBin]), 1);
                            ;
                        }
                    }
                }
            }
        }
    }
}
