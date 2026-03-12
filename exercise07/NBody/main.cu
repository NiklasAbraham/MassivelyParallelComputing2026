#include "gltools.h"
#include "Tools.h"

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#if USE_GL_INTEROP
#include <cuda_gl_interop.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace std;

#define GUI
#define NUM_FRAMES 250

#define THREADS_PER_BLOCK 128
#define EPS_2 0.00001f
#define GRAVITY 0.00000001f
#define WORLD_MIN -2.0f
#define WORLD_MAX 2.0f
#define CUTOFF_RADIUS 0.06f
#define COARSE_GRID_DIM 48
#define FAR_SKIP_FACTOR 1.1f

// Set to 0 to disable CUDA-GL interop (required for VNC / software GL).
#define USE_GL_INTEROP 0

#if USE_GL_INTEROP
struct cudaGraphicsResource* cudaGLPositions;
#endif

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float) randI / (float) RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}

inline __device__ float2 operator+(const float2 op1, const float2 op2)
{
    return make_float2(op1.x + op2.x, op1.y + op2.y);
}

inline __device__ float2 operator-(const float2 op1, const float2 op2)
{
    return make_float2(op1.x - op2.x, op1.y - op2.y);
}

inline __device__ float2 operator*(const float2 op1, const float op2)
{
    return make_float2(op1.x * op2, op1.y * op2);
}

inline __device__ float2 operator/(const float2 op1, const float op2)
{
    return make_float2(op1.x / op2, op1.y / op2);
}

inline __device__ void operator+=(float2 &a, const float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __device__ float2 calculateAcceleration(const float2 subjectPos,
        const float2 otherPos, const float otherMass)
{
    float2 direction = otherPos - subjectPos;
    float directionSqLen = direction.x * direction.x
            + direction.y * direction.y;

    float denominator = directionSqLen + EPS_2;
    denominator = rsqrtf(denominator * denominator * denominator);

    return direction * otherMass * denominator;
}

// Part 1: Baseline O(N^2) all-pairs force computation.
__global__ void updateAccelerationsBruteforce(float2* accelerations,
        const float2* positions, const float* masses,
        const unsigned int numBodies)
{
    // MISSING: Implement the brute-force N-body kernel.
    // One thread should compute the acceleration for one body by summing the
    // interaction with all other bodies via `calculateAcceleration(...)`.
    // Note: `calculateAcceleration(...)` already implements one pairwise
    // contribution. The factor `GRAVITY` is applied later during integration.

    unsigned int bodyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;

    float2 total_accl = make_float2(0.f, 0.f);
    for (int i =0; i < numBodies; ++i){
        // if (i != bodyIdx) // the equation does not specify this??????
        // actually we can add this safely because the diff of positions will be 0
        total_accl += calculateAcceleration(positions[bodyIdx], positions[i],
                                            masses[i]);
    }

    accelerations[bodyIdx] = total_accl;
}

inline __device__ int clampCellCoord(const int coord, const int gridDim)
{
    return max(0, min(coord, gridDim - 1));
}

inline __device__ int positionToCellIdx(const float2 position,
        const float worldMin, const float cellSize, const int gridDim)
{
    int cellX = clampCellCoord((int) ((position.x - worldMin) / cellSize),
            gridDim);
    int cellY = clampCellCoord((int) ((position.y - worldMin) / cellSize),
            gridDim);
    return cellY * gridDim + cellX;
}

__global__ void buildBodyCellPairs(int* bodyCellIndices, int* bodyIndices,
        const float2* positions, const unsigned int numBodies,
        const float worldMin, const float cellSize, const int gridDim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies)
        return;

    bodyCellIndices[idx] = positionToCellIdx(positions[idx], worldMin, cellSize,
            gridDim);
    bodyIndices[idx] = idx;
}

__global__ void resetCellRanges(int* cellStarts, int* cellEnds,
        const int numCells)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int) idx >= numCells)
        return;
    cellStarts[idx] = -1;
    cellEnds[idx] = -1;
}

__global__ void resetCoarseGrid(float* coarseMasses, float2* coarseWeightedCom,
        const int numCoarseCells)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int) idx >= numCoarseCells)
        return;
    coarseMasses[idx] = 0.0f;
    coarseWeightedCom[idx] = make_float2(0.0f, 0.0f);
}

__global__ void accumulateCoarseGrid(const float2* positions, const float* masses,
        float* coarseMasses, float2* coarseWeightedCom,
        const unsigned int numBodies, const float worldMin,
        const float coarseCellSize, const int coarseGridDim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies)
        return;

    int coarseIdx = positionToCellIdx(positions[idx], worldMin, coarseCellSize,
            coarseGridDim);
    float mass = masses[idx];
    atomicAdd(&coarseMasses[coarseIdx], mass);
    atomicAdd(&coarseWeightedCom[coarseIdx].x, positions[idx].x * mass);
    atomicAdd(&coarseWeightedCom[coarseIdx].y, positions[idx].y * mass);
}

__global__ void finalizeCoarseGrid(const float* coarseMasses,
        const float2* coarseWeightedCom, float2* coarseComs,
        const int numCoarseCells)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int) idx >= numCoarseCells)
        return;
    float mass = coarseMasses[idx];
    if (mass > 0.0f)
        coarseComs[idx] = coarseWeightedCom[idx] / mass;
    else
        coarseComs[idx] = make_float2(0.0f, 0.0f);
}

inline __device__ float pointAabbDistSq(const float2 p, const float2 minP,
        const float2 maxP)
{
    float dx = 0.0f;
    if (p.x < minP.x)
        dx = minP.x - p.x;
    else if (p.x > maxP.x)
        dx = p.x - maxP.x;

    float dy = 0.0f;
    if (p.y < minP.y)
        dy = minP.y - p.y;
    else if (p.y > maxP.y)
        dy = p.y - maxP.y;

    return dx * dx + dy * dy;
}

__global__ void buildCellRanges(const int* sortedCellIndices, int* cellStarts,
        int* cellEnds, const unsigned int numBodies)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies)
        return;

    int cell = sortedCellIndices[idx];
    if (idx == 0 || cell != sortedCellIndices[idx - 1])
        cellStarts[cell] = idx;
    if (idx == numBodies - 1 || cell != sortedCellIndices[idx + 1])
        cellEnds[cell] = idx + 1;
}

// Part 2: Uniform-grid acceleration structure with near/far hybrid approximation.
// The fine grid is used for exact interactions in the local neighborhood, while
// the coarse grid summarizes far-away regions by total mass and center of mass.
__global__ void updateAccelerationsGridCutoff(float2* accelerations,
        const float2* positions, const float* masses,
        const int* sortedBodyIndices,
        const int* cellStarts, const int* cellEnds, const unsigned int numBodies,
        const float worldMin, const float cellSize, const int gridDim,
        const float cutoffSq, const float* coarseMasses,
        const float2* coarseComs, const int coarseGridDim,
        const float coarseCellSize)
{
    // MISSING: Implement the grid-based N-body kernel.
    // Combine exact interactions from nearby fine-grid cells with a far-field
    // approximation based on the coarse-grid centers of mass.
    // `sortedBodyIndices` maps from the sorted cell order back to the original
    // body index. `cellStarts` / `cellEnds` store the half-open range
    // [start, end) of bodies belonging to each fine-grid cell.
    // All of the constance that you need are defined as macros above.

    unsigned int bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bodyIdx >= numBodies) return;
    
    //sorted idx

    float2 myPosition = positions[bodyIdx];
    int myCell = positionToCellIdx(myPosition, worldMin, cellSize, gridDim);
    // unsigned int mySortedIdx = sortedBodyIndices[bodyIdx];

    // myCellIdx = myCelly * gridDim + myCellx
    int myCellX = myCell % gridDim;
    int myCellY = myCell / gridDim;

    float2 total_acc = make_float2(0.f, 0.f);

    //add accl from everything around my cell
    // for (int idx = cellStarts[myCell]; idx < cellEnds[myCell];  idx++){
    //     total_acc += calculateAcceleration(myPosition, positions[sortedBodyIndices[idx]],
    //                                        masses[sortedBodyIndices[idx]]);
    // }

    for (int dy = -1; dy <= 1; ++dy){
        for (int dx = -1; dx <= 1; ++dx){
            int nx = clampCellCoord(myCellX + dx, gridDim);
            int ny = clampCellCoord(myCellY + dy, gridDim);

            int neighborCell = ny * gridDim + nx;
            int start = cellStarts[neighborCell];
            int end = cellEnds[neighborCell];
            if (start == -1) continue;

            for (int i =start; i < end; ++i){
                int otherIdx = sortedBodyIndices[i];
                total_acc += calculateAcceleration(myPosition, positions[otherIdx],
                                                    masses[otherIdx]);
            }
        }
    }

    // now the coarse grid
    for (int cy =0; cy < coarseGridDim; ++cy){
        for (int cx =0; cx < coarseGridDim; ++cx){
            int coarseIdx = cy * coarseGridDim + cx;

            if (coarseMasses[coarseIdx] == 0.0f) continue;

            float2 cellMin = make_float2(worldMin + cx * coarseCellSize, 
                                         worldMin + cy * coarseCellSize);

            float2 cellMax = make_float2(cellMin.x + coarseCellSize, 
                                         cellMin.y + coarseCellSize);
            

            float distSq = pointAabbDistSq(myPosition, cellMin, cellMax);

            if (distSq > cutoffSq * FAR_SKIP_FACTOR){
                total_acc += calculateAcceleration(myPosition, coarseComs[coarseIdx], coarseMasses[coarseIdx]);
            }
        }
    }

    accelerations[bodyIdx] = total_acc;

}

__global__ void updateVelocitiesPositions(float2* positions, float2* velocities,
        const float2* accelerations, const unsigned int numBodies)
{
    // MISSING: Integrate one simulation step.
    // Update each body's velocity using the computed acceleration and then
    // advance the position by the updated velocity.
    // The intended update is explicit Euler:
    //   v <- v + GRAVITY * a
    //   p <- p + v

    unsigned int bodyIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (bodyIdx >= numBodies) return;

    float2 oldVel = velocities[bodyIdx];
    // float2 * float is defined but float * float2 is not...
    float2 newVel = oldVel +  accelerations[bodyIdx] * GRAVITY;
    velocities[bodyIdx] = newVel;
    positions[bodyIdx] += newVel;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0]
                << " <numBodies> [bruteforce|grid] [--no-gui]" << endl;
        return 1;
    }

    unsigned int numBodies = atoi(argv[1]);
    unsigned int numBlocks = (numBodies + THREADS_PER_BLOCK - 1)
            / THREADS_PER_BLOCK;
    enum Mode
    {
        BRUTEFORCE,
        GRID
    };
    Mode mode = GRID;
    bool enableGui = true;
    for (int argIdx = 2; argIdx < argc; argIdx++)
    {
        string arg = argv[argIdx];
        if (arg == "bruteforce")
            mode = BRUTEFORCE;
        else if (arg == "grid")
            mode = GRID;
        else if (arg == "--no-gui")
            enableGui = false;
        else
        {
            cout << "Unknown argument '" << arg << "'." << endl;
            cout << "Usage: " << argv[0]
                    << " <numBodies> [bruteforce|grid] [--no-gui]"
                    << endl;
            return 1;
        }
    }
    const char* modeName = mode == BRUTEFORCE ? "bruteforce" : "grid-hybrid";
    cout << "Simulation mode: " << modeName << endl;
    cout << "GUI: " << (enableGui ? "on" : "off") << endl;

    float2* hPositions = new float2[numBodies];
    float2* hVelocities = new float2[numBodies];
    float* hMasses = new float[numBodies];

    float2* gPositions = nullptr;
    float2* gVelocities = nullptr;
    float2* gAccelerations = nullptr;
    float* gMasses = nullptr;
    int* gBodyCellIndices = nullptr;
    int* gBodyIndices = nullptr;
    int* gCellStarts = nullptr;
    int* gCellEnds = nullptr;
    float* gCoarseMasses = nullptr;
    float2* gCoarseWeightedCom = nullptr;
    float2* gCoarseComs = nullptr;

    const float cellSize = CUTOFF_RADIUS;
    const float cutoffSq = CUTOFF_RADIUS * CUTOFF_RADIUS;
    const int gridDim = (int) ((WORLD_MAX - WORLD_MIN) / cellSize) + 1;
    const int numCells = gridDim * gridDim;
    const unsigned int numCellBlocks = (numCells + THREADS_PER_BLOCK - 1)
            / THREADS_PER_BLOCK;
    const float coarseCellSize = (WORLD_MAX - WORLD_MIN)
            / (float) COARSE_GRID_DIM;
    const int numCoarseCells = COARSE_GRID_DIM * COARSE_GRID_DIM;
    const unsigned int numCoarseBlocks = (numCoarseCells + THREADS_PER_BLOCK - 1)
            / THREADS_PER_BLOCK;

    cudaMalloc(&gPositions, numBodies * sizeof(float2));
    cudaMalloc(&gVelocities, numBodies * sizeof(float2));
    cudaMalloc(&gAccelerations, numBodies * sizeof(float2));
    cudaMalloc(&gMasses, numBodies * sizeof(float));
    cudaMalloc(&gBodyCellIndices, numBodies * sizeof(int));
    cudaMalloc(&gBodyIndices, numBodies * sizeof(int));
    cudaMalloc(&gCellStarts, numCells * sizeof(int));
    cudaMalloc(&gCellEnds, numCells * sizeof(int));
    cudaMalloc(&gCoarseMasses, numCoarseCells * sizeof(float));
    cudaMalloc(&gCoarseWeightedCom, numCoarseCells * sizeof(float2));
    cudaMalloc(&gCoarseComs, numCoarseCells * sizeof(float2));

    for (unsigned int i = 0; i < numBodies; i++)
    {
        hPositions[i].x = randF(-1.0, 1.0);
        hPositions[i].y = randF(-1.0, 1.0);
        hVelocities[i].x = hPositions[i].y * 0.007f + randF(0.001f, -0.001f);
        hVelocities[i].y = -hPositions[i].x * 0.007f + randF(0.001f, -0.001f);
        hMasses[i] = randF(0.0f, 1.0f) * 10000.0f / (float) numBodies;
    }

    cudaMemcpy(gPositions, hPositions, numBodies * sizeof(float2),
            cudaMemcpyHostToDevice);
    cudaMemcpy(gVelocities, hVelocities, numBodies * sizeof(float2),
            cudaMemcpyHostToDevice);
    cudaMemcpy(gMasses, hMasses, numBodies * sizeof(float),
            cudaMemcpyHostToDevice);

    delete[] hVelocities;
    delete[] hMasses;

#ifdef GUI
    GLuint sp = 0;
    GLuint vb = 0;
    GLuint va = 0;
    if (enableGui)
    {
        initGL();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        sp = createShaderProgram(SHADER_DIR "white.vs", 0, 0, 0, SHADER_DIR "white.fs");

        glGenBuffers(1, &vb);
        GL_CHECK_ERROR;
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        GL_CHECK_ERROR;
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, 0,
                GL_STREAM_DRAW);
        GL_CHECK_ERROR;
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        GL_CHECK_ERROR;

#if USE_GL_INTEROP
        cudaGraphicsGLRegisterBuffer(&cudaGLPositions, vb,
                cudaGraphicsMapFlagsWriteDiscard);
#endif

        glGenVertexArrays(1, &va);
        GL_CHECK_ERROR;
        glBindVertexArray(va);
        GL_CHECK_ERROR;
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        GL_CHECK_ERROR;
        glEnableVertexAttribArray(glGetAttribLocation(sp, "inPosition"));
        GL_CHECK_ERROR;
        glVertexAttribPointer(glGetAttribLocation(sp, "inPosition"), 2, GL_FLOAT,
                GL_FALSE, 0, 0);
        GL_CHECK_ERROR;
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        GL_CHECK_ERROR;
        glBindVertexArray(0);
        GL_CHECK_ERROR;
    }
#endif

    for (unsigned int t = 0; t < NUM_FRAMES; t++)
    {
        __int64_t frameStart = continuousTimeNs();
        __int64_t accelKernelNs = 0;
        __int64_t integrateKernelNs = 0;
        __int64_t renderNs = 0;

        __int64_t stepStart = continuousTimeNs();
        if (mode == BRUTEFORCE)
        {
            // MISSING: Launch the brute-force acceleration kernel.
            // Reuse the launch configuration that is already computed above and
            // pass the buffers needed to write accelerations from all bodies.
        updateAccelerationsBruteforce<<<numBlocks, THREADS_PER_BLOCK>>>(gAccelerations, gPositions, gMasses, numBodies);
        }
        else
        {
            // MISSING: Launch the grid-based acceleration workflow.
            // Hint: you first need a body-to-cell mapping, then a way to group
            // bodies by cell, then metadata for occupied cell ranges, and
            // finally the data required for the far-field approximation before
            // computing accelerations.
            //
            // A useful way to think about the existing helper kernels is:
            // 1. assign each body to one fine-grid cell
            // 2. sort bodies by fine-grid cell index (you can use thrust::sort_by_key() to do this)
            // 3. build [start, end) ranges for each occupied fine-grid cell
            // 4. accumulate total mass and weighted positions per coarse cell (the kernels already exist in the code)
            // 5. divide by mass to get each coarse cell's center of mass (the kernels already exist in the code)
            // 6. launch the hybrid near-field / far-field acceleration kernel
            //
            // Buffer roles:
            // - `gBodyCellIndices`: fine-grid cell index per body
            // - `gBodyIndices`: body indices reordered by sorted cell index
            // - `gCellStarts`, `gCellEnds`: ranges into the sorted arrays
            // - `gCoarseMasses`, `gCoarseWeightedCom`, `gCoarseComs`:
            //   temporary and final data for the far-field approximation
            //
            // The launch sizes and all required parameters are already prepared
            // in the surrounding code.

            buildBodyCellPairs<<<numBlocks, THREADS_PER_BLOCK>>>(gBodyCellIndices, gBodyIndices, gPositions,
                                                                 numBodies, WORLD_MIN, cellSize, gridDim);

            /*
            example:
            before sorting:
            bodyCellIndices: [5, 2, 5, 1]
            bodyIndices:     [0, 1, 2, 3]

            after sorting:
            gBodyCellIndices: [1, 2, 5, 5]
            gBodyIndices:     [3, 1, 0, 2]
            */
            thrust::device_ptr<int> keys(gBodyCellIndices);
            thrust::device_ptr<int> values(gBodyIndices);
            thrust::sort_by_key(keys, keys + numBodies, values);
            
            resetCellRanges<<<numCellBlocks, THREADS_PER_BLOCK>>>(gCellStarts, gCellEnds, numCells);

            buildCellRanges<<<numBlocks, THREADS_PER_BLOCK>>>(gBodyCellIndices, gCellStarts, gCellEnds, numBodies);

            resetCoarseGrid<<<numCoarseBlocks, THREADS_PER_BLOCK>>>(gCoarseMasses, gCoarseWeightedCom, numCoarseCells);
            
            accumulateCoarseGrid<<<numCoarseBlocks, THREADS_PER_BLOCK>>>(gPositions, gMasses, gCoarseMasses, gCoarseWeightedCom, 
                                                                numBodies, WORLD_MIN, coarseCellSize, COARSE_GRID_DIM);
            
            finalizeCoarseGrid<<<numCoarseBlocks, THREADS_PER_BLOCK>>>(gCoarseMasses, gCoarseWeightedCom, gCoarseComs, numCoarseCells);

            updateAccelerationsGridCutoff<<<numBlocks, THREADS_PER_BLOCK>>>(gAccelerations, gPositions, gMasses, gBodyIndices, gCellStarts,
                                                                        gCellEnds, numBodies, WORLD_MIN, cellSize, gridDim, cutoffSq, gCoarseMasses, 
                                                                        gCoarseComs, COARSE_GRID_DIM, coarseCellSize);
        }
        updateVelocitiesPositions<<<numBlocks, THREADS_PER_BLOCK>>>(gPositions, gVelocities, gAccelerations, numBodies);
        cudaDeviceSynchronize();
        accelKernelNs = continuousTimeNs() - stepStart;

        stepStart = continuousTimeNs();
        // MISSING: Launch the integration kernel
        // Use the same general launch setup as above and pass the state arrays
        // needed to update motion for all bodies.
        cudaDeviceSynchronize();
        integrateKernelNs = continuousTimeNs() - stepStart;

#ifdef GUI
        if (enableGui)
        {
            __int64_t renderStart = continuousTimeNs();
#if USE_GL_INTEROP
            cudaGraphicsMapResources(1, &cudaGLPositions);
            float2* glPositions;
            size_t num_bytes;
            cudaGraphicsResourceGetMappedPointer((void**)&glPositions,
                    &num_bytes, cudaGLPositions);
            cudaMemcpy(glPositions, gPositions, numBodies * sizeof(float2),
                    cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &cudaGLPositions);
#else
            cudaMemcpy(hPositions, gPositions, numBodies * sizeof(float2),
                    cudaMemcpyDeviceToHost);
            glBindBuffer(GL_ARRAY_BUFFER, vb);
            glBufferSubData(GL_ARRAY_BUFFER, 0,
                    numBodies * sizeof(float2), hPositions);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(sp);
            glBindVertexArray(va);
            glDrawArrays(GL_POINTS, 0, numBodies);
            glBindVertexArray(0);
            glUseProgram(0);
            swapBuffers();
            renderNs = continuousTimeNs() - renderStart;
        }
#endif

        __int64_t frameTotalNs = continuousTimeNs() - frameStart;
        cout << "Frame total: " << frameTotalNs
                << "ns | accel kernel (" << modeName << "): " << accelKernelNs
                << "ns | integration kernel: " << integrateKernelNs
                << "ns | render: " << renderNs << "ns" << endl;
    }

#ifdef GUI
    if (enableGui)
    {
        cout << "Done." << endl;
        sleep(2);
        glDeleteProgram(sp);
        GL_CHECK_ERROR;
        glDeleteVertexArrays(1, &va);
        GL_CHECK_ERROR;

#if USE_GL_INTEROP
        cudaGraphicsUnregisterResource(cudaGLPositions);
#endif

        glDeleteBuffers(1, &vb);
        GL_CHECK_ERROR;
        exitGL();
    }
#endif

    cudaFree(gPositions);
    cudaFree(gVelocities);
    cudaFree(gAccelerations);
    cudaFree(gMasses);
    cudaFree(gBodyCellIndices);
    cudaFree(gBodyIndices);
    cudaFree(gCellStarts);
    cudaFree(gCellEnds);
    cudaFree(gCoarseMasses);
    cudaFree(gCoarseWeightedCom);
    cudaFree(gCoarseComs);

    delete[] hPositions;
}
