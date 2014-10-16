/**
 *  @file Partition.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef PARTITION_H_
#define PARTITION_H_

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace mass {

class Partition {

public:

	/**
	 *  Returns the number of elements in this partition.
	 */
	virtual int size() = 0;

	/**
	 *  Returns the number of place elements and ghost elements.
	 */
	int sizePlusGhosts() {
		return numElements;
	}

	/**
	 *  Gets the rank of this partition.
	 */
	int getRank() {
		return rank;
	}

	/**
	 *  Returns an array of the Place elements contained in this Partition object. This is an expensive
	 *  operation since it requires memory transfer.
	 */
	void *hostPtr() {
		void *retVal = hPtr;
		if (rank > 0) {
			retVal += ghostWidth;
		}
		return retVal;
	}

	/**
	 *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
	 */
	void *hostPtrPlusGhosts() {
		return hPtr;
	}

	/**
	 *  Returns the pointer to the GPU data. NULL if not on GPU.
	 */
	void *devicePtr() {
		return dPtr;
	}

	void setDevicePtr(void *places)) {
		dPtr = places;
	}

	/**
	 *  Returns the handle associated with this Partition object that was set at construction.
	 */
	int getHandle() {
		return handle;
	}

	/**
	 *  Sets the start and number of places in this partition.
	 */
	void setSection(void *start) {
		hPtr = start;
	}

	void setQty(int qty) {
		numElements = qty;
		setIdealDims();
	}

	bool isLoaded() {
		return isloaded;
	}

	void makeLoadable() {
		if (!loadable) {
			if (dPtr != NULL) {
				cudaFree(dPtr);
			}

			cudaMalloc((void**) &dPtr, Tbytes * sizePlusGhosts());
			loadable = true;
		}
	}

	void *load(cudaStream_t stream) {
		makeLoadable();

		cudaMemcpyAsync(dPtr, hPtr, Tbytes * sizePlusGhosts(),
				cudaMemcpyHostToDevice, stream);
		loaded = true;
	}

	bool retrieve(cudaStream_t stream, bool freeOnRetrieve) {
		bool retreived = loaded;

		if (loaded) {
			cudaMemcpyAsync(hPtr, dPtr, Tbytes * sizePlusGhosts(),
					cudaMemcpyDeviceToHost, stream);
			loaded = false;
		}

		if (freeOnRetreive) {
			cudaFree(dPtr);
			loadable = false;
			dPtr = NULL;
		}

		return retreived;
	}

	int getGhostWidth() {
		return ghostWidth;
	}

	void setGhostWidth(int width, int n, int *dimensions) {
		ghostWidth = width;

		// start at 1 because we never want to factor in x step
		for (int i = 1; i < n; ++i) {
			ghostWidth += dimensions[i];
		}
	}

	void updateLeftGhost(void *ghost, cudaStream_t stream) {
		if (rank > 0) {
			if (isloaded) {
				cudaMemcpyAsync(dPtr, ghost, Tbytes * ghostWidth,
						cudaMemcpyHostToDevice, stream);
			} else {
				memcpy(hPtr, ghost, Tbytes * ghostWidth);
			}
		}
	}

	void updateRightGhost(void *ghost, cudaStream_t stream) {
		if (rank < Partition::numRanks - 1) {
			if (isloaded) {
				cudaMemcpyAsync(dPtr + numElements, ghost, Tbytes * ghostWidth,
						cudaMemcpyHostToDevice, stream);
			} else {
				memcpy(hPtr + ghostWidth + numElements, ghost,
						Tbytes * ghostWidth);
			}
		}
	}

	void *getLeftBuffer() {
		if (isloaded) {
			cudaMemcpy(hPtr, dPtr + ghostWidth, Tbytes * ghostWidth,
					cudaMemcpyDeviceToHost);
		}

		return hPtr + ghostWidth;
	}

	void *getRightBuffer()() {
		if(isloaded) {
			cudaMemcpy( hPtr, dPtr + numElements, Tbytes * ghostWidth, cudaMemcpyDeviceToHost );
		}
		return hPtr + numElements;
	}

	dim3 blockDim() {
		return dims[0];
	}

	dim3 threadDim() {
		return dims[1];
	}

	void setIdealDims() {
		int numBlocks = (numElements - 1) / THREADS_PER_BLOCK + 1;
		dim3 blockDim(numBlocks, 1, 1);

		int nThr = (numElements - 1) / numBlocks + 1;
		dim3 threadDim(nThr, 1, 1);

		dims[0] = blockDim;
		dims[1] = threadDim;
	}

	int getPlaceBytes() {
		return Tbytes;
	}

private:
	void *hPtr; // this starts at the left ghost, and extends to the end of the right ghost
	void *dPtr; // pointer to GPU data
	static int numRanks; // the overall number of ranks in this model
	int handle;         // User-defined identifier for this Partition
	int rank; // the rank of this partition
	int numElements;    // the number of place elements in this Partition
	int Tbytes; // sizeof(agent)
	bool isloaded;
	bool loadable;
	int ghostWidth;
	dim3 dims[2]; // 0 is blockdim, 1 is threaddim
};

} /* namespace mass */
#endif // PARTITION_H_