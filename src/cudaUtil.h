/*! \file cudaUtil.h
 * Rob Jordan

 * Utility functions for CUDA applications
 *
 * Error checking functions were adapted from
 * http://choorucode.wordpress.com/2011/03/02/cuda-error-checking/
 * and "Cuda by Example" book.h sample code.
 *
 * define CERR to turn on error checking
 */
#pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

namespace mass {

#define BLOCK_SIZE 256  // max threads per block
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define CERR

/*! If a cuda error occurs, terminates the program with a descriptive error message.
 */
#define CATCH(err) __cudaCatch( err, __FILE__, __LINE__ )

/*! Tests whether a cuda error has occured and terminates the program with a descriptive error
 * message if so.
 */
#define CHECK() __cudaCheckError( __FILE__, __LINE__ )

/*!
 * Tests if a random number generation error has occurred and terminates the program with a descriptive error.
 */
#define CURAND_CATCH(err) __curandCatch(err, __FILE__, __LINE__)

/*! Terminates the program with a descriptive error message if a cuda error occurs.
 */
void __cudaCatch(cudaError err, const char *file, const int line);

void __curandCatch(curandStatus_t err, const char *file, const int line);

/*! Tests whether a cuda error has occured and terminates the program with a descriptive error
 * message if so.
 */
void __cudaCheckError(const char *file, const int line);

/*! More careful checking. However, this will affect performance.*/
void __cudaCheckSync(const char *file, const int line);

/*! A combination of calloc() and cudaMalloc(). Allocates device memory and sets the memory to zero.
 * Performed asynchronously.
 */
cudaError_t cudaCallocAsync(void **devPtr, size_t size, cudaStream_t stream);

/******************************************************************************
 * These global indexing helper functions were created and published by Martin
 * Peniak. See http://www.martinpeniak.com/index.php?option=com_
 * content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
 * for the original functions.
 *****************************************************************************/
// 1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D();

// 1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D();

// 1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D();

// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D();

// 2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D();

// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D();

// 3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D();

// 3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D();

// 3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D();

} /* namespace mass */
