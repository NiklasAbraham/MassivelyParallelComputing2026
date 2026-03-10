#include "SignalStrengthsSortedCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"

#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// "Smart" CUDA implementation which computes signal strengths
//
// First, all transmitters are sorted into buckets
// Then, all receivers are sorted into buckets
// Then, receivers only compute signal strength against transmitters in nearby buckets
//
// This multi-step algorithm makes the signal strength computation scale much
//  better to high number of transmitters/receivers

struct Bucket
{
    int startIndex; // Start of bucket within array
    int numElements; // Number of elements in bucket
};

// Domain is [0,1] x [0,1]; bucket (bx, by) covers x in [bx/N, (bx+1)/N), y in [by/N, (by+1)/N]
static __device__ int getBucketIndex(const Position& p)
{
    int bx = (int)(p.x * BucketsPerAxis);
    int by = (int)(p.y * BucketsPerAxis);
    if (bx >= (int)BucketsPerAxis)
        bx = BucketsPerAxis - 1;
    if (by >= (int)BucketsPerAxis)
        by = BucketsPerAxis - 1;
    return by * BucketsPerAxis + bx;
}

// Scan for bucket sizes to build histogram (count per bucket)
static __global__ void countBucketsKernel(const Position* inputPositions, int numInputPositions,
                                         int* histogram)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numInputPositions)
        return;
    int bucket = getBucketIndex(inputPositions[i]);
    atomicAdd(&histogram[bucket], 1);
}

// Exclusive scan of histogram to determine bucket start indices and fill Bucket structs
static __global__ void computeBucketStartsKernel(const int* histogram, Bucket* outputBuckets)
{
    __shared__ int s_hist[256];
    int i = threadIdx.x;
    int numBuckets = BucketsPerAxis * BucketsPerAxis;
    if (i < numBuckets)
        s_hist[i] = histogram[i];

    __syncthreads();

    // Single-thread exclusive scan over s_hist (one block, 256 threads)
    if (i == 0)
    {
        int sum = 0;
        for (int k = 0; k < numBuckets; ++k)
        {
            outputBuckets[k].startIndex = sum;
            outputBuckets[k].numElements = s_hist[k];
            sum += s_hist[k];
        }
    }
}

// Copy bucket start indices into scatterOffsets for use in scatter (atomicAdd target)
static __global__ void initScatterOffsetsKernel(const Bucket* outputBuckets, int* scatterOffsets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int numBuckets = BucketsPerAxis * BucketsPerAxis;
    if (i < numBuckets)
        scatterOffsets[i] = outputBuckets[i].startIndex;
}

// Scatter positions into output by bucket (each bucket contiguous)
static __global__ void scatterToBucketsKernel(const Position* inputPositions, int numInputPositions,
                                              Position* outputPositions, int* scatterOffsets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numInputPositions)
        return;
    const Position& p = inputPositions[i];
    int bucket = getBucketIndex(p);
    int outIdx = atomicAdd(&scatterOffsets[bucket], 1);
    outputPositions[outIdx] = p;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Sort a set of positions into a set of buckets
//
// Given a set of input positions, these will be re-ordered such that
//  each range of elements in the output array belong to the same bucket.
// The list of buckets that is output describes where each such range begins
//  and ends in the re-ordered position array.

static void sortPositionsIntoBuckets(CudaArray<Position>& cudaInputPositions,
                                     CudaArray<Position>& cudaOutputPositions,
                                     CudaArray<Bucket>& cudaOutputPositionBuckets)
{
    // Bucket sorting with "Counting Sort" is a multi-phase process:
    //
    // 1. Determine how many of the input elements should end up in each bucket (build a histogram)
    // 2. Given the histogram, compute where in the output array that each bucket begins (prefix sum)
    // 3. Scatter elements from the input array into the output array by bucket

    int numInputPositions = cudaInputPositions.size();
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Step 1: Scan for bucket sizes (histogram)
    CudaArray<int> cudaHistogram(numBuckets);
    cudaMemset(cudaHistogram.cudaArray(), 0, numBuckets * sizeof(int));

    int blockSize = 256;
    int numBlocks = (numInputPositions + blockSize - 1) / blockSize;
    countBucketsKernel<<<numBlocks, blockSize>>>(cudaInputPositions.cudaArray(), numInputPositions,
                                                 cudaHistogram.cudaArray());

    // Step 2: Calculate buckets' start positions (exclusive scan) and fill Bucket structs
    computeBucketStartsKernel<<<1, 256>>>(cudaHistogram.cudaArray(),
                                          cudaOutputPositionBuckets.cudaArray());

    // Step 3: Sort positions into buckets (scatter using start indices + atomic offsets)
    CudaArray<int> cudaScatterOffsets(numBuckets);
    initScatterOffsetsKernel<<<1, 256>>>(cudaOutputPositionBuckets.cudaArray(),
                                         cudaScatterOffsets.cudaArray());

    scatterToBucketsKernel<<<numBlocks, blockSize>>>(
        cudaInputPositions.cudaArray(), numInputPositions, cudaOutputPositions.cudaArray(),
        cudaScatterOffsets.cudaArray());
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Go through all transmitters in one bucket, find highest signal strength
// Return highest strength (or the old value, if that was higher)

static __device__ float scanBucket(const Position* transmitters, int numTransmitters,
                                   const Position& receiver, float bestSignalStrength)
{
    for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
    {
        const Position& transmitter = transmitters[transmitterIndex];

        float strength = signalStrength(transmitter, receiver);

        if (bestSignalStrength < strength)
            bestSignalStrength = strength;
    }

    return bestSignalStrength;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculate signal strength for all receivers

static __global__ void calculateSignalStrengthsSortedKernel(const Position* transmitters,
                                                            const Bucket* transmitterBuckets,
                                                            const Position* receivers,
                                                            const Bucket* receiverBuckets,
                                                            float* signalStrengths)
{
    // Determine which bucket the current grid block is processing

    int receiverBucketIndexX = blockIdx.x;
    int receiverBucketIndexY = blockIdx.y;

    int receiverBucketIndex = receiverBucketIndexY * BucketsPerAxis + receiverBucketIndexX;

    const Bucket& receiverBucket = receiverBuckets[receiverBucketIndex];

    int receiverStartIndex = receiverBucket.startIndex;
    int numReceivers = receiverBucket.numElements;

    // Distribute available receivers over the set of available threads

    for (int receiverIndex = threadIdx.x; receiverIndex < numReceivers; receiverIndex += blockDim.x)
    {
        // Locate current receiver within the current bucket

        const Position& receiver = receivers[receiverStartIndex + receiverIndex];
        float& finalStrength = signalStrengths[receiverStartIndex + receiverIndex];

        float bestSignalStrength = 0.f;

        // Scan all buckets in the 3x3 region enclosing the receiver's bucket index

        for (int transmitterBucketIndexY = receiverBucketIndexY - 1;
             transmitterBucketIndexY < receiverBucketIndexY + 2; ++transmitterBucketIndexY)
            for (int transmitterBucketIndexX = receiverBucketIndexX - 1;
                 transmitterBucketIndexX < receiverBucketIndexX + 2; ++transmitterBucketIndexX)
            {
                // Only process bucket if its index is within [0, BucketsPerAxis - 1] along each
                // axis

                if (transmitterBucketIndexX >= 0 && transmitterBucketIndexX < BucketsPerAxis
                    && transmitterBucketIndexY >= 0 && transmitterBucketIndexY < BucketsPerAxis)
                {
                    // Scan bucket for a potential new "highest signal strength"

                    int transmitterBucketIndex =
                        transmitterBucketIndexY * BucketsPerAxis + transmitterBucketIndexX;
                    int transmitterStartIndex =
                        transmitterBuckets[transmitterBucketIndex].startIndex;
                    int numTransmitters = transmitterBuckets[transmitterBucketIndex].numElements;
                    bestSignalStrength = scanBucket(&transmitters[transmitterStartIndex],
                                                    numTransmitters, receiver, bestSignalStrength);
                }
            }

        // Store out the highest signal strength found for the receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy input positions to device memory

    CudaArray<Position> cudaTempTransmitters(cpuTransmitters.size());
    cudaTempTransmitters.copyToCuda(&(*cpuTransmitters.begin()));

    CudaArray<Position> cudaTempReceivers(cpuReceivers.size());
    cudaTempReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Allocate device memory for sorted arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Bucket> cudaTransmitterBuckets(numBuckets);

    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<Bucket> cudaReceiverBuckets(numBuckets);

    // Sort transmitters and receivers into buckets

    sortPositionsIntoBuckets(cudaTempTransmitters, cudaTransmitters, cudaTransmitterBuckets);
    sortPositionsIntoBuckets(cudaTempReceivers, cudaReceivers, cudaReceiverBuckets);

    // Perform signal strength computation
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);

    calculateSignalStrengthsSortedKernel<<<grid, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitterBuckets.cudaArray(), cudaReceivers.cudaArray(),
        cudaReceiverBuckets.cudaArray(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory
    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}
