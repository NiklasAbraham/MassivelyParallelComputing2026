#include "Tools.h"

#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// #define VERBOSE // Prints input matrix and results. Only uncomment for small matrix sizes!
#define RUN_CPU // Runs CPU code for reference (slow!!!)
#define N 1000 // Must be a multiple of THREADS_PER_BLOCK
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * n + col];
        }
        result[row * n + col] = sum;
    }
}

__global__ void multiplyMatrixGpuSharedMemory(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a more sophisticated GPU square matrix multiplication.
    // Compute square submatrices per block. Load the common input
    // data of all threads of a block into shared memory cooperatively.
    int const TileSize = THREADS_PER_BLOCK;
    __shared__ float a_shared[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ float b_shared[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    
    for (int k = 0; k < (n + TileSize - 1) / TileSize; k++)
    {
        int a_col = k * TileSize + threadIdx.y;
        int b_row = k * TileSize + threadIdx.x;

        if (row < n && a_col < n)
            a_shared[threadIdx.x][threadIdx.y] = a[row * n + a_col];
        else
            a_shared[threadIdx.x][threadIdx.y] = 0.0f;
        if (b_row < n && col < n)
            b_shared[threadIdx.x][threadIdx.y] = b[b_row * n + col];
        else
            b_shared[threadIdx.x][threadIdx.y] = 0.0f;
        __syncthreads();

        // compute the result
        for (int k = 0; k < TileSize; k++)
        {
            sum += a_shared[threadIdx.x][k] * b_shared[k][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < n && col < n)
    {
        result[row * n + col] = sum;
    }
}

int main(int argc, char** argv)
{
    __int64_t startTime;
    __int64_t endTime;

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
