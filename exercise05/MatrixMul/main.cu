#include "Tools.h"

#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// #define VERBOSE // Prints input matrix and results. Only uncomment for small matrix sizes!
// #define RUN_CPU // Runs CPU code for reference (slow!!!)
#define N 4096 // Must be a multiple of THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 32 // per axis -> block has this value squared threads.
void multiplyMatrix(float* result, const float* a, const float* b, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            result[i * n + j] = 0.0f;
            for (unsigned int k = 0; k < n; k++)
            {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void dumpMatrix(const float* m, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << setw(3) << setprecision(3) << m[i * n + j] << " ";
        }
        cout << endl;
    }
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}

__global__ void multiplyMatrixGpuBasic(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a trivial GPU square matrix multiplication.
    // Use one thread per output element.
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    float res = 0.f;
    for (int k =0; k < n; ++k){
        res += a[i * n + k] * b[k * n + j];
    }
    result[i * n + j] = res;
}

/*
helpful stuff:
    https://youtu.be/Q3GgbfGTnVc
*/
__global__ void multiplyMatrixGpuSharedMemory(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a more sophisticated GPU square matrix multiplication.
    // Compute square submatrices per block. Load the common input
    // data of all threads of a block into shared memory cooperatively.

    __shared__ float shared_a[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ float shared_b[THREADS_PER_BLOCK][THREADS_PER_BLOCK];

    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= n || j >= n) return;

    float res = 0.f;
    //we know that THREADS_PER_BLOCK | n because of the comments above
    for (int tile = 0; tile < (n / THREADS_PER_BLOCK); ++tile){
        shared_a[threadIdx.y][threadIdx.x] = a[i * n + tile*THREADS_PER_BLOCK + threadIdx.x];
        shared_b[threadIdx.y][threadIdx.x] = b[(tile*THREADS_PER_BLOCK + threadIdx.y) * n + j];

        __syncthreads();
        // at this point the tile has been loaded into shared memory

        // now we compute partial dot product and add it to the result
        for (int k =0; k < THREADS_PER_BLOCK; ++k)
            res += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        
        // why this syncthreads is needed?
        // to ensure all threads have read the shared mem before we rewrite it for new blocks
        __syncthreads();
    }
    result[i * n + j] = res;
}

int main(int argc, char** argv)
{
    int64_t startTime;
    int64_t endTime;

    // Allocate all memory
    float* hM1 = new float[N * N];
    float* hM2 = new float[N * N];
    float* hMR = new float[N * N];
    float* gM1;
    cudaMalloc(&gM1, sizeof(float) * N * N);
    float* gM2;
    cudaMalloc(&gM2, sizeof(float) * N * N);
    float* gMR;
    cudaMalloc(&gMR, sizeof(float) * N * N);

    // Initialize matrices and upload to CUDA
    for (unsigned int n = 0; n < N * N; n++)
    {
        hM1[n] = randF(-1.0, 1.0);
        hM2[n] = randF(-1.0, 1.0);
    }
    cudaMemcpy(gM1, hM1, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gM2, hM2, sizeof(int) * N * N, cudaMemcpyHostToDevice);
#ifdef VERBOSE
    cout << "Input Matrices:" << endl;
    dumpMatrix(hM1, N);
    cout << endl;
    dumpMatrix(hM2, N);
    cout << endl << endl;
#endif

#ifdef RUN_CPU
    // Calculations on CPU
    startTime = continuousTimeNs();
    multiplyMatrix(hMR, hM1, hM2, N);
    endTime = continuousTimeNs();
#ifdef VERBOSE
    cout << "CPU:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "CPU time: " << (endTime - startTime) << "ns" << endl;
#endif

    // Calculations on GPU
    int blocksPerGridX =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    int blocksPerGridY =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    startTime = continuousTimeNs();
    multiplyMatrixGpuBasic<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU simple:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "GPU simple time: " << (endTime - startTime) << "ns" << endl;
    startTime = continuousTimeNs();
    multiplyMatrixGpuSharedMemory<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU advanced:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "GPU advanced time: " << (endTime - startTime) << "ns" << endl;

    // Free all memory
    cudaFree(gM1);
    cudaFree(gM2);
    cudaFree(gMR);
    delete[] hM1;
    delete[] hM2;
    delete[] hMR;

    checkCUDAError("end of program");
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
