struct Dim3 {
    uniform int x, y, z;
};
uniform float fx = 1.0f;
uniform float fy = 1.0f;
uniform float fz = 1.0f;
uniform const int mx = 64;
uniform const int my = 64;
uniform const int mz = 64;
uniform const int sPencils = 4;
uniform const int lPencils = 32;
uniform float c_ax;
uniform float c_bx;
uniform float c_cx;
uniform float c_dx;
uniform float c_ay;
uniform float c_by;
uniform float c_cy;
uniform float c_dy;
uniform float c_az;
uniform float c_bz;
uniform float c_cz;
uniform float c_dz;
uniform;
export void derivative_x(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                         uniform unsigned int32 shared_memory_size,
                         uniform float s_f[sPencils][mx + 8], uniform float f[],
                         uniform float df[]) {
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
                            if (blockIdx.x == 0 && threadIdx.x == 0) {
                            }
                            int i = threadIdx.x;
                            int j = blockIdx.x * blockDim.y + threadIdx.y;
                            int k = blockIdx.y;
                            int si = i + 4;
                            int sj = threadIdx.y;
                            int globalIdx = k * mx * my + j * mx + i;
                            s_f[sj][si] = f[globalIdx];
                        }
                    }
                }

                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                         threadIdx.y++) {
                        for (threadIdx.x = programIndex;
                             threadIdx.x < blockDim.x;
                             threadIdx.x += programCount) {
                            int i = threadIdx.x;
                            int j = blockIdx.x * blockDim.y + threadIdx.y;
                            int k = blockIdx.y;
                            int si = i + 4;
                            int sj = threadIdx.y;
                            int globalIdx = k * mx * my + j * mx + i;
                            if (i < 4) {
                                s_f[sj][si - 4] = s_f[sj][si + mx - 5];
                                s_f[sj][si + mx] = s_f[sj][si + 1];
                            }
                        }
                    }
                }

                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                         threadIdx.y++) {
                        for (threadIdx.x = programIndex;
                             threadIdx.x < blockDim.x;
                             threadIdx.x += programCount) {
                            int i = threadIdx.x;
                            int j = blockIdx.x * blockDim.y + threadIdx.y;
                            int k = blockIdx.y;
                            int si = i + 4;
                            int sj = threadIdx.y;
                            int globalIdx = k * mx * my + j * mx + i;
                            df[globalIdx] =
                                (c_ax * (s_f[sj][si + 1] - s_f[sj][si - 1]) +
                                 c_bx * (s_f[sj][si + 2] - s_f[sj][si - 2]) +
                                 c_cx * (s_f[sj][si + 3] - s_f[sj][si - 3]) +
                                 c_dx * (s_f[sj][si + 4] - s_f[sj][si - 4]));
                        }
                    }
                }
            }
        }
    }
}
