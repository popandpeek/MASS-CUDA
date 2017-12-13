/* cudaUtil.cu
 * Rob Jordan
 */

#include "cudaUtil.h"
#include "stdio.h"     // fprintf()
#include "Mass.h"
#include "MassException.h"
#include "Logger.h"
namespace mass {

void __cudaCatch(cudaError err, const char *file, const int line) {
#ifdef CERR
	if (cudaSuccess != err) {
		fprintf(stderr, "MASS Cuda Util: %s in %s at line %d\n",
				cudaGetErrorString(err), file, line);
		Logger::error("MASS Cuda Util: %s in %s at line %d\n",
				cudaGetErrorString(err), file, line);
		Mass::finish();
		throw MassException("An error occured ");
	}
#endif
}

void __curandCatch(curandStatus_t err, const char *file, const int line) {
#ifdef CERR
	if (err != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "MASS Cuda Util: cuRand error in %s at line %d\n", file, line);
		Logger::error("MASS Cuda Util: cuRand error in %s at line %d\n", file, line);
		Mass::finish();
		throw MassException("An error occured ");
	}
#endif
}

void __cudaCheckError(const char *file, const int line) {
#ifdef CERR
	__cudaCatch(cudaGetLastError(), file, line);
#endif
}

void __cudaCheckSync(const char *file, const int line) {
#ifdef CERR
	__cudaCatch(cudaGetLastError(), file, line);
	__cudaCatch(cudaDeviceSynchronize(), file, line);
#endif
}

cudaError_t cudaCallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
	cudaError_t err = cudaMalloc(devPtr, size);
	if (err == cudaSuccess) {
		err = cudaMemsetAsync(*devPtr, 0, size, stream);
	}
	return err;
}

/******************************************************************************
 * These global indexing helper functions were created and published by Martin
 * Peniak. See http://www.martinpeniak.com/index.php?option=com_
 * content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
 * for the original functions.
 *****************************************************************************/
// 1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

// 1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D() {
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x
			+ threadIdx.x;
}

// 1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D() {
	return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x;
}

// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D() {
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

// 2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

// 3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

// 3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

// 3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

} /* namespace mass */
