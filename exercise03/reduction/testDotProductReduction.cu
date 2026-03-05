// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009
//
//   Ulm University
//
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_BLOCKS 256
#define MAX_THREADS 256

inline __int64_t continuousTimeNs()
{
    timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    __int64_t result = (__int64_t)now.tv_sec * 1000000000 + (__int64_t)now.tv_nsec;

    return result;
}

__global__ void dotProdKernel(float* dst, const float* a1, const float* a2, int dim)
{
    // Number of the current thread
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    // Number of all threads
    unsigned int threadSize = gridDim.x * blockDim.x;

    // Sum up every (threadSize)th element starting at the threads index and ending before dim
    float result = 0.0f;
    for (unsigned int t = threadNo; t < dim; t += threadSize)
        result += a1[t] * a2[t];

    // Write the result to dst[threadIdx] if it can contain something
    dst[threadNo] = result;
}


__global__ void reduceKernel(float* dst, const float* src, int dim)
{

    // from the slides in version 2
    __shared__ float volatile partialSum[MAX_THREADS];

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + t;

    // load input into shared memory (or 0 if out of range)
    partialSum[t] = (i < dim) ? src[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) // step down by 2
    {
        __syncthreads();

        if (t < stride)
            partialSum[t] += partialSum[t+stride];
    }
    __syncthreads();

    /*
    // unroll last 6 predicated steps
    // simple first solution
    if (t <= 32) {
        partialSum[t] += partialSum[t + 32];
        partialSum[t] += partialSum[t + 16];
        partialSum[t] += partialSum[t + 8];
        partialSum[t] += partialSum[t + 4];
        partialSum[t] += partialSum[t + 2];
        partialSum[t] += partialSum[t + 1];
    }
    __syncthreads();

    if (t == 0)
        dst[blockIdx.x] = partialSum[0];
    */
    float value = partialSum[t];
    for (int i=1; i<32; i*=2)
        value += __shfl_xor_sync(0xffffffffu, value, i);

    if (t == 0)
        dst[blockIdx.x] = value;
}



/* This program sets up two large arrays of size dim and computes the
 dot product of both arrays.

 Most of the code of previous exercises is reused.
 Mode 0 of the program computes the final dot product as before.

 Mode 1: After computing the dot product and storing the result for all
 MAX_BLOCKS * MAX_THREAD threads, this time, the reduction of the sum
 is to be computed on the GPU.

 Write a reduction sum kernel which reduces the input to the sum in log(n) steps.
 The number of total threads will be divided by nThreads(iter-1) in each iteration.

 Inside the kernel, the problem will be reduced by a factor of 2 in each step.

 */

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;

    if (argc < 3)
    {
        printf("usage: testDotProductStreams <dim> <reduction mode [CPU only:0, CPU sum:1, GPU reduction:2]>\n");
        exit(1);
    }

    // number of elements in both vectors
    int dim = atoi(argv[acount++]);

    int mode = atoi(argv[acount++]);

    printf("dim: %d\n", dim);

    // Allocate only pagelocked memory for simplicity
    float* cpuArray1;
    float* cpuArray2;
    float* cpuResult;
    cudaMallocHost((void**)&cpuArray1, dim * sizeof(float));
    cudaMallocHost((void**)&cpuArray2, dim * sizeof(float));
    cudaMallocHost((void**)&cpuResult, MAX_THREADS * MAX_BLOCKS * sizeof(float));

    // initialize the two arrays
    for (int i = 0; i < dim; ++i)
    {
#ifdef RTEST
        cpuArray1[i] = drand48();
        cpuArray2[i] = drand48();
#else
        cpuArray1[i] = 1.0;
        cpuArray2[i] = 1.; // i % 10;
#endif
    }

    // Allocate GPU memory
    float* gpuArray1;
    float* gpuArray2;
    float* gpuResult1; // Two result arrays to be able to move data from one to the other during
                       // reduction
    float* gpuResult2;
    cudaMalloc((void**)&gpuArray1, dim * sizeof(float));
    cudaMalloc((void**)&gpuArray2, dim * sizeof(float));

    cudaMalloc((void**)&gpuResult1, MAX_BLOCKS * MAX_THREADS * sizeof(float));
    cudaMalloc((void**)&gpuResult2, MAX_BLOCKS * MAX_THREADS * sizeof(float));
    
    float* gpuReducedResult;
    cudaMalloc((void**)&gpuReducedResult, MAX_BLOCKS * sizeof(float));

    // Upload input data

    cudaMemcpy(gpuArray1, cpuArray1, dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpuArray2, cpuArray2, dim * sizeof(float), cudaMemcpyHostToDevice);

    // Variable for output
    double finalDotProduct = 0.;

    __int64_t startTime = continuousTimeNs();

    // Iterations for benchmarking only the kernel call
    for (int iter = 0; iter < 1000; ++iter)
    {
        // a simplistic way of splitting the problem into threads
        dim3 blockGrid(MAX_BLOCKS);
        dim3 threadBlock(MAX_THREADS);

        unsigned int expectedResultSize;

        switch (mode)
        {
        case 0:
            finalDotProduct = 0.0;

            for (unsigned int i = 0; i < dim; i++)
                finalDotProduct += cpuArray1[i] * cpuArray2[i];

            break;

        case 1:
            // call the dot kernel
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult1, gpuArray1, gpuArray2, dim);

            // If dim < launchedThreads, only the first dim elements will contain data
            expectedResultSize = min(dim, MAX_THREADS * MAX_BLOCKS);

            // download and combine the results of multiple threads

            cudaMemcpy(cpuResult, gpuResult1, expectedResultSize * sizeof(float),
                       cudaMemcpyDeviceToHost);

            finalDotProduct = 0.;

            // accumulate the final result on the host
            for (int i = 0; i < expectedResultSize; ++i)
                finalDotProduct += cpuResult[i];

            break;

        case 2:
            // call the dot kernel, store result in gpuResult1
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult1, gpuArray1, gpuArray2, dim);

            // reduce over partial results: first dim elements are valid (rest are 0)
            expectedResultSize = (unsigned int)min(dim, MAX_THREADS * MAX_BLOCKS);
            reduceKernel<<<blockGrid, threadBlock>>>(gpuResult2, gpuResult1, expectedResultSize);
            reduceKernel<<<1, MAX_THREADS>>>(gpuReducedResult, gpuResult2, MAX_BLOCKS);
            {
                float reducedFloat = 0.f;
                cudaMemcpy(&reducedFloat, gpuReducedResult, sizeof(float), cudaMemcpyDeviceToHost);
                finalDotProduct = (double)reducedFloat;
            }
            break;

        } // end switch
    }

    __int64_t endTime = continuousTimeNs();
    __int64_t runTime = endTime - startTime;

    // Print results and timing
    printf("Result: %f\n", finalDotProduct);
    printf("Time: %f\n", (float)runTime / 1000000000.0f);

    // cleanup GPU memory
    cudaFree(gpuReducedResult);
    cudaFree(gpuResult1);
    cudaFree(gpuResult2);
    cudaFree(gpuArray2);
    cudaFree(gpuArray1);

    // free page locked memory
    cudaFreeHost(cpuArray1);
    cudaFreeHost(cpuArray2);
    cudaFreeHost(cpuResult);

    checkCUDAError("end of program");

    printf("done\n");
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
