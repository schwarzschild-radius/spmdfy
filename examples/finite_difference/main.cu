#include "cuda_utils.cuh"
#include "finite_difference.cu"
#include "finite_difference_ispc.h"
#include <iostream>

// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to
//     calculate the derivative at mutiple points

dim3 grid[3][2], block[3][2];
ispc::ConstantMemory cm;

// host routine to set constant data
void setDerivativeParameters() {
    // check to make sure dimensions are integral multiples of sPencils
    if ((mx % sPencils != 0) || (my % sPencils != 0) || (mz % sPencils != 0)) {
        printf("'mx', 'my', and 'mz' must be integral multiples of sPencils\n");
        exit(1);
    }

    if ((mx % lPencils != 0) || (my % lPencils != 0)) {
        printf("'mx' and 'my' must be multiples of lPencils\n");
        exit(1);
    }

    // stencil weights (for unit length problem)
    float dsinv = mx - 1.f;

    float ax = 4.f / 5.f * dsinv;
    float bx = -1.f / 5.f * dsinv;
    float cx = 4.f / 105.f * dsinv;
    float dx = -1.f / 280.f * dsinv;
    cm.c_ax = ax, cm.c_bx = bx, cm.c_cx = cx, cm.c_dx = dx;
    cudaCheck(cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_bx, &bx, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_cx, &cx, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_dx, &dx, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));

    dsinv = my - 1.f;

    float ay = 4.f / 5.f * dsinv;
    float by = -1.f / 5.f * dsinv;
    float cy = 4.f / 105.f * dsinv;
    float dy = -1.f / 280.f * dsinv;
    cm.c_ay = ay, cm.c_by = by, cm.c_cy = cy, cm.c_dy = dy;
    cudaCheck(cudaMemcpyToSymbol(c_ay, &ay, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_by, &by, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_cy, &cy, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_dy, &dy, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));

    dsinv = mz - 1.f;

    float az = 4.f / 5.f * dsinv;
    float bz = -1.f / 5.f * dsinv;
    float cz = 4.f / 105.f * dsinv;
    float dz = -1.f / 280.f * dsinv;
    cm.c_az = az, cm.c_bz = bz, cm.c_cz = cz, cm.c_dz = dz;
    cudaCheck(cudaMemcpyToSymbol(c_az, &az, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_bz, &bz, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_cz, &cz, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyToSymbol(c_dz, &dz, sizeof(float), 0,
                                 cudaMemcpyHostToDevice));

    // Execution configurations for small and large pencil tiles

    grid[0][0] = dim3(my / sPencils, mz, 1);
    block[0][0] = dim3(mx, sPencils, 1);

    grid[0][1] = dim3(my / lPencils, mz, 1);
    block[0][1] = dim3(mx, sPencils, 1);

    grid[1][0] = dim3(mx / sPencils, mz, 1);
    block[1][0] = dim3(sPencils, my, 1);

    grid[1][1] = dim3(mx / lPencils, mz, 1);
    // we want to use the same number of threads as above,
    // so when we use lPencils instead of sPencils in one
    // dimension, we multiply the other by sPencils/lPencils
    block[1][1] = dim3(lPencils, my * sPencils / lPencils, 1);

    grid[2][0] = dim3(mx / sPencils, my, 1);
    block[2][0] = dim3(sPencils, mz, 1);

    grid[2][1] = dim3(mx / lPencils, my, 1);
    block[2][1] = dim3(lPencils, mz * sPencils / lPencils, 1);
}

void initInput(float *f, int dim) {
    const float twopi = 8.f * (float)atan(1.0);

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                switch (dim) {
                case 0:
                    f[k * mx * my + j * mx + i] =
                        cos(fx * twopi * (i - 1.f) / (mx - 1.f));
                    break;
                case 1:
                    f[k * mx * my + j * mx + i] =
                        cos(fy * twopi * (j - 1.f) / (my - 1.f));
                    break;
                case 2:
                    f[k * mx * my + j * mx + i] =
                        cos(fz * twopi * (k - 1.f) / (mz - 1.f));
                    break;
                }
            }
        }
    }
}

void initSol(float *sol, int dim) {
    const float twopi = 8.f * (float)atan(1.0);

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                switch (dim) {
                case 0:
                    sol[k * mx * my + j * mx + i] =
                        -fx * twopi * sin(fx * twopi * (i - 1.f) / (mx - 1.f));
                    break;
                case 1:
                    sol[k * mx * my + j * mx + i] =
                        -fy * twopi * sin(fy * twopi * (j - 1.f) / (my - 1.f));
                    break;
                case 2:
                    sol[k * mx * my + j * mx + i] =
                        -fz * twopi * sin(fz * twopi * (k - 1.f) / (mz - 1.f));
                    break;
                }
            }
        }
    }
}

void checkResults(double &error, double &maxError, float *sol, float *df) {
    // error = sqrt(sum((sol-df)**2)/(mx*my*mz))
    // maxError = maxval(abs(sol-df))
    maxError = 0;
    error = 0;
    size_t mismatch_count = 0;
    size_t total_count = 0;
    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
            for (int i = 0; i < mx; i++) {
                float s = sol[k * mx * my + j * mx + i];
                float f = df[k * mx * my + j * mx + i];
                if ((int)(s * 1000) != (int)(f * 1000)) {
                    mismatch_count++;
                    // printf("%d %d %d: %f %f\n", i, j, k, s, f);
                }
                total_count++;
                error += (s - f) * (s - f);
                if (fabs(s - f) > maxError)
                    maxError = fabs(s - f);
            }
        }
    }
    std::cout << "total count: " << total_count << "\n";
    std::cout << "MisMathed count: " << mismatch_count << "\n";
    error = sqrt(error / (mx * my * mz));
}

// void printResults(float *cuda_f, float *ispc_f){

// }

// Run the kernels for a given dimension. One for sPencils, one for lPencils
void runTest(int dimension) {
    void (*fpDeriv[2])(float *, float *);
    void (*ispcDeriv[2])(const ispc::Dim3 &, const ispc::Dim3 &,
                         const unsigned int shared_memory_size, float *,
                         float *, const int32_t, const int32_t, const int32_t);
    float *ispc_sm_ptr[2];
    const int pencil[2] = {sPencils, lPencils};
    switch (dimension) {
    case 0:
        fpDeriv[0] = derivative_x;
        fpDeriv[1] = derivative_x_lPencils;
        ispcDeriv[0] = ispc::derivative_x;
        ispcDeriv[1] = ispc::derivative_x_lPencils;
        ispc_sm_ptr[0] = new float[sPencils * (mx + 8)];
        ispc_sm_ptr[1] = new float[lPencils * (mx + 8)];
        break;
    case 1:
        fpDeriv[0] = derivative_y;
        fpDeriv[1] = derivative_y_lPencils;
        ispcDeriv[0] = ispc::derivative_y;
        ispcDeriv[1] = ispc::derivative_y_lPencils;
        ispc_sm_ptr[0] = new float[sPencils * (my + 8)];
        ispc_sm_ptr[1] = new float[lPencils * (my + 8)];
        break;
    case 2:
        fpDeriv[0] = derivative_z;
        fpDeriv[1] = derivative_z_lPencils;
        ispcDeriv[0] = ispc::derivative_z;
        ispcDeriv[1] = ispc::derivative_z_lPencils;
        ispc_sm_ptr[0] = new float[sPencils * (mz + 8)];
        ispc_sm_ptr[1] = new float[lPencils * (mz + 8)];
        break;
    }

    int sharedDims[3][2][2] = {mx,       sPencils, mx,       lPencils,
                               sPencils, my,       lPencils, my,
                               sPencils, mz,       lPencils, mz};

    float *f = new float[mx * my * mz];
    float *df = new float[mx * my * mz];
    float *ispc_f = new float[mx * my * mz];
    float *sol = new float[mx * my * mz];

    initInput(f, dimension);
    initSol(sol, dimension);

    // device arrays
    int bytes = mx * my * mz * sizeof(float);
    float *d_f, *d_df;
    cudaCheck(cudaMalloc((void **)&d_f, bytes));
    cudaCheck(cudaMalloc((void **)&d_df, bytes));

    const int nReps = 20;
    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    cudaCheck(cudaEventCreate(&startEvent));
    cudaCheck(cudaEventCreate(&stopEvent));

    double error, maxError;

    printf("%c derivatives\n\n", (char)(0x58 + dimension));

    for (int fp = 0; fp < 2; fp++) {
        cudaCheck(cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(d_df, 0, bytes));
        memset(ispc_f, 0, bytes);

        fpDeriv[fp]<<<grid[dimension][fp], block[dimension][fp]>>>(
            d_f, d_df); // warm up
        cudaCheck(cudaEventRecord(startEvent, 0));
        for (int i = 0; i < nReps; i++) {
            fpDeriv[fp]<<<grid[dimension][fp], block[dimension][fp]>>>(d_f,
                                                                       d_df);
            ispcDeriv[fp]({grid[dimension][fp].x, grid[dimension][fp].y,
                           grid[dimension][fp].z},
                          {block[dimension][fp].x, block[dimension][fp].y,
                           block[dimension][fp].z},
                          f, ispc_f, pencil[fp], mx, my);
        }

        cudaCheck(cudaEventRecord(stopEvent, 0));
        cudaCheck(cudaEventSynchronize(stopEvent));
        cudaCheck(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

        cudaCheck(cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost));

        checkResults(error, maxError, sol, df);

        printf("  Using shared memory tile of %d x %d\n",
               sharedDims[dimension][fp][0], sharedDims[dimension][fp][1]);
        printf("   RMS error: %e\n", error);
        printf("   MAX error: %e\n", maxError);
        printf("   Average time (ms): %f\n", milliseconds / nReps);
        printf("   Average Bandwidth (GB/s): %f\n\n",
               2.f * 1e-6 * mx * my * mz * nReps * sizeof(float) /
                   milliseconds);
        checkResults(error, maxError, sol, ispc_f);
        printf("   RMS error: %e\n", error);
        printf("   MAX error: %e\n", maxError);
    }

    cudaCheck(cudaEventDestroy(startEvent));
    cudaCheck(cudaEventDestroy(stopEvent));

    cudaCheck(cudaFree(d_f));
    cudaCheck(cudaFree(d_df));

    delete[] f;
    delete[] df;
    delete[] sol;
    delete[] ispc_f;
    delete[] ispc_sm_ptr[0];
    delete[] ispc_sm_ptr[1];
}

// This the main host code for the finite difference
// example.  The kernels are contained in the derivative_m module

int main(void) {
    // Print device and precision
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    setDerivativeParameters(); // initialize

    runTest(0); // x derivative
    runTest(1); // y derivative
    // runTest(2); // z derivative

    return 0;
}