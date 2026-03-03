// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
//
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "PPM.hh"

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_THREADS 128

//-------------------------------------------------------------------------------

// compute absolute difference between two images
__global__ void diffKernel(float* _dst, const float* _src1, const float* _src2, int _w)
{
    int x = blockIdx.x * MAX_THREADS + threadIdx.x;
    int y = blockIdx.y;
    int pos = y * _w + x;

    if (x < _w)
    {
        float v1 = _src1[pos];
        float v2 = _src2[pos];
        _dst[pos] = fabsf(v1 - v2);
    }
}

//-------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    int acount = 1; // parse command line

    if (argc < 4)
    {
        printf("usage: %s <inImg1> <inImg2> <outImg>\n", argv[0]);
        exit(1);
    }

    float* img1;
    float* img2;

    bool success = true;
    int w1, h1, w2, h2;
    success &= readPPM(argv[acount++], w1, h1, &img1);
    success &= readPPM(argv[acount++], w2, h2, &img2);
    if (!success)
    {
        fprintf(stderr, "Error reading input images.\n");
        exit(1);
    }
    if (w1 != w2 || h1 != h2)
    {
        fprintf(stderr, "Input images must have the same size.\n");
        exit(1);
    }

    int w = w1;
    int h = h1;
    int nPix = w * h;

    float* gpuImg1;
    float* gpuImg2;
    float* gpuResImg;

    //-------------------------------------------------------------------------------
    printf("Executing the GPU Version\n");

    cudaMalloc((void**)&gpuImg1, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuImg2, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuResImg, nPix * 3 * sizeof(float));

    cudaMemcpy(gpuImg1, img1, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuImg2, img2, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // calculate the block dimensions
    dim3 threadBlock(MAX_THREADS);
    // select the number of blocks vertically (*3 because of RGB)
    dim3 blockGrid((w * 3 + MAX_THREADS - 1) / MAX_THREADS, h, 1);
    printf("bl/thr: %d  %d %d\n", blockGrid.x, blockGrid.y, threadBlock.x);

    diffKernel<<<blockGrid, threadBlock>>>(gpuResImg, gpuImg1, gpuImg2, w * 3);

    checkCUDAError("kernel execution");

    // download result
    cudaMemcpy(img1, gpuResImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpuResImg);
    cudaFree(gpuImg1);
    cudaFree(gpuImg2);

    writePPM(argv[acount++], w, h, img1);

    delete[] img1;
    delete[] img2;

    checkCUDAError("end of program");

    printf("  done\n");
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
