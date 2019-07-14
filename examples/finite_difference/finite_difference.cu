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

    //   if(blockIdx.x == 0 && threadIdx.x == 0){
    //       printf("%f, %f, %f, %f\n", c_ax, c_bx, c_cx, c_dx);
    //   }

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

__global__ void derivative_x_lPencils(float *f, float *df) {
    __shared__ float s_f[lPencils][mx + 8]; // 4-wide halo

    int i = threadIdx.x;
    int jBase = blockIdx.x * lPencils;
    int k = blockIdx.y;
    int si = i + 4; // local i for shared memory access + halo offset

    for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
        int globalIdx = k * mx * my + (jBase + sj) * mx + i;
        s_f[sj][si] = f[globalIdx];
    }

    __syncthreads();

    // fill in periodic images in shared memory array
    if (i < 4) {
        for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
            s_f[sj][si - 4] = s_f[sj][si + mx - 5];
            s_f[sj][si + mx] = s_f[sj][si + 1];
        }
    }

    __syncthreads();

    for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
        int globalIdx = k * mx * my + (jBase + sj) * mx + i;
        df[globalIdx] = (c_ax * (s_f[sj][si + 1] - s_f[sj][si - 1]) +
                         c_bx * (s_f[sj][si + 2] - s_f[sj][si - 2]) +
                         c_cx * (s_f[sj][si + 3] - s_f[sj][si - 3]) +
                         c_dx * (s_f[sj][si + 4] - s_f[sj][si - 4]));
    }
}

__global__ void derivative_y(float *f, float *df) {
    __shared__ float s_f[my + 8][sPencils];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;
    int si = threadIdx.x;
    int sj = j + 4;

    int globalIdx = k * mx * my + j * mx + i;

    s_f[sj][si] = f[globalIdx];

    __syncthreads();

    if (j < 4) {
        s_f[sj - 4][si] = s_f[sj + my - 5][si];
        s_f[sj + my][si] = s_f[sj + 1][si];
    }

    __syncthreads();

    df[globalIdx] = (c_ay * (s_f[sj + 1][si] - s_f[sj - 1][si]) +
                     c_by * (s_f[sj + 2][si] - s_f[sj - 2][si]) +
                     c_cy * (s_f[sj + 3][si] - s_f[sj - 3][si]) +
                     c_dy * (s_f[sj + 4][si] - s_f[sj - 4][si]));
}

__global__ void derivative_y_lPencils(float *f, float *df) {
    __shared__ float s_f[my + 8][lPencils];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int si = threadIdx.x;

    for (int j = threadIdx.y; j < my; j += blockDim.y) {
        int globalIdx = k * mx * my + j * mx + i;
        int sj = j + 4;
        s_f[sj][si] = f[globalIdx];
    }

    __syncthreads();

    int sj = threadIdx.y + 4;
    if (sj < 8) {
        s_f[sj - 4][si] = s_f[sj + my - 5][si];
        s_f[sj + my][si] = s_f[sj + 1][si];
    }

    __syncthreads();

    for (int j = threadIdx.y; j < my; j += blockDim.y) {
        int globalIdx = k * mx * my + j * mx + i;
        int sj = j + 4;
        df[globalIdx] = (c_ay * (s_f[sj + 1][si] - s_f[sj - 1][si]) +
                         c_by * (s_f[sj + 2][si] - s_f[sj - 2][si]) +
                         c_cy * (s_f[sj + 3][si] - s_f[sj - 3][si]) +
                         c_dy * (s_f[sj + 4][si] - s_f[sj - 4][si]));
    }
}

__global__ void derivative_z(float *f, float *df) {
    __shared__ float s_f[mz + 8][sPencils];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.y;
    int si = threadIdx.x;
    int sk = k + 4; // halo offset

    int globalIdx = k * mx * my + j * mx + i;

    s_f[sk][si] = f[globalIdx];

    __syncthreads();

    if (k < 4) {
        s_f[sk - 4][si] = s_f[sk + mz - 5][si];
        s_f[sk + mz][si] = s_f[sk + 1][si];
    }

    __syncthreads();

    df[globalIdx] = (c_az * (s_f[sk + 1][si] - s_f[sk - 1][si]) +
                     c_bz * (s_f[sk + 2][si] - s_f[sk - 2][si]) +
                     c_cz * (s_f[sk + 3][si] - s_f[sk - 3][si]) +
                     c_dz * (s_f[sk + 4][si] - s_f[sk - 4][si]));
}

__global__ void derivative_z_lPencils(float *f, float *df) {
    __shared__ float s_f[mz + 8][lPencils];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int si = threadIdx.x;

    for (int k = threadIdx.y; k < mz; k += blockDim.y) {
        int globalIdx = k * mx * my + j * mx + i;
        int sk = k + 4;
        s_f[sk][si] = f[globalIdx];
    }

    __syncthreads();

    int k = threadIdx.y + 4;
    if (k < 8) {
        s_f[k - 4][si] = s_f[k + mz - 5][si];
        s_f[k + mz][si] = s_f[k + 1][si];
    }

    __syncthreads();

    for (int k = threadIdx.y; k < mz; k += blockDim.y) {
        int globalIdx = k * mx * my + j * mx + i;
        int sk = k + 4;
        df[globalIdx] = (c_az * (s_f[sk + 1][si] - s_f[sk - 1][si]) +
                         c_bz * (s_f[sk + 2][si] - s_f[sk - 2][si]) +
                         c_cz * (s_f[sk + 3][si] - s_f[sk - 3][si]) +
                         c_dz * (s_f[sk + 4][si] - s_f[sk - 4][si]));
    }
}