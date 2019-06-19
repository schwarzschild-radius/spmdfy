#include <cuda_runtime.h>

float fx = 1.0f;
float fy = 1.0f;
float fz = 1.0f;
const int mx = 64;
const int my = 64;
const int mz = 64;

const int sPencils = 4;
const int lPencils = 32;

__constant__ float c_ax;
__constant__ float c_bx;
__constant__ float c_cx;
__constant__ float c_dx;
__constant__ float c_ay;
__constant__ float c_by;
__constant__ float c_cy;
__constant__ float c_dy;
__constant__ float c_az;
__constant__ float c_bz;
__constant__ float c_cz;
__constant__ float c_dz;

__global__ void derivative_x(float *f, float *df) {
    __shared__ float s_f[sPencils][mx + 8]; // 4-wide halo

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("%f, %f, %f, %f\n", c_ax, c_bx, c_cx, c_dx);
    }

    int i = threadIdx.x;
    int j = blockIdx.x * blockDim.y + threadIdx.y;
    int k = blockIdx.y;
    int si = i + 4;       // local i for shared memory access + halo offset
    int sj = threadIdx.y; // local j for shared memory access

    int globalIdx = k * mx * my + j * mx + i;

    s_f[sj][si] = f[globalIdx];

    __syncthreads();

    // fill in periodic images in shared memory array
    if (i < 4) {
        s_f[sj][si - 4] = s_f[sj][si + mx - 5];
        s_f[sj][si + mx] = s_f[sj][si + 1];
    }

    __syncthreads();

    df[globalIdx] = (c_ax * (s_f[sj][si + 1] - s_f[sj][si - 1]) +
                     c_bx * (s_f[sj][si + 2] - s_f[sj][si - 2]) +
                     c_cx * (s_f[sj][si + 3] - s_f[sj][si - 3]) +
                     c_dx * (s_f[sj][si + 4] - s_f[sj][si - 4]));
}