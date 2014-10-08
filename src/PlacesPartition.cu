/**
 *  @file PlacesPartition.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#define THREADS_PER_BLOCK 512

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "Dispatcher.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Mass.h"

namespace mass {

inline void *shiftPtr(void *origin, int qty, int Tsize) {
	char *retVal = (char*) origin;
	retVal += Tsize * qty;
	return retVal;
}

PlacesPartition::PlacesPartition(int handle, int rank, int numElements,
		int ghostWidth, int n, int *dimensions, int Tsize) :
		hPtr(NULL), dPtr(NULL), handle(handle), rank(rank), numElements(
				numElements), isloaded(false), Tsize(Tsize) {
	setGhostWidth(ghostWidth, n, dimensions);
	setIdealDims();
}

/**
 *  Destructor
 */
PlacesPartition::~PlacesPartition() {
}

/**
 *  Returns the number of place elements in this partition.
 */
int PlacesPartition::size() {
	return numElements;
}

/**
 *  Returns the number of place elements and ghost elements.
 */
int PlacesPartition::sizePlusGhosts() {
	int numRanks = Mass::getPlaces(handle)->getNumPartitions();
	if (1 == numRanks) {
		return numElements;
	}

	int retVal = numElements;
	if (0 == rank || numRanks - 1 == rank) {
		// there is only one ghost width on an edge rank
		retVal += ghostWidth;
	} else {
		retVal += 2 * ghostWidth;
	}

	return retVal;
}

/**
 *  Gets the rank of this partition.
 */
int PlacesPartition::getRank() {
	return rank;
}

/**
 *  Returns an array of the Place elements contained in this PlacesPartition object. This is an expensive
 *  operation since it requires memory transfer.
 */
void *PlacesPartition::hostPtr() {
	void *retVal = hPtr;
	if (rank > 0) {
		retVal = shiftPtr(hPtr, ghostWidth, Tsize);
	}
	return retVal;
}

/**
 *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
 */
void *PlacesPartition::hostPtrPlusGhosts() {
	return hPtr;
}

/**
 *  Returns the pointer to the GPU data. NULL if not on GPU.
 */
void *PlacesPartition::devicePtr() {
	return dPtr;
}

void PlacesPartition::setDevicePtr(void *places) {
	dPtr = places;
}

/**
 *  Returns the handle associated with this PlacesPartition object that was set at construction.
 */
int PlacesPartition::getHandle() {
	return handle;
}

bool PlacesPartition::isLoaded() {
	return isloaded;
}

void PlacesPartition::setLoaded(bool loaded) {
	isloaded = loaded;
}

int PlacesPartition::getGhostWidth() {
	return ghostWidth;
}

void PlacesPartition::setGhostWidth(int width, int n, int *dimensions) {
	ghostWidth = width;

	// start at 1 because we never want to factor in x step
	for (int i = 1; i < n; ++i) {
		ghostWidth += dimensions[i];
	}

	// set pointer
	Places *places = Mass::getPlaces(handle);
	if (0 == rank) {
		hPtr = places->elements;
	} else {
		hPtr = shiftPtr(places->elements, rank * numElements - ghostWidth, Tsize);
	}
}

void *PlacesPartition::getLeftBuffer() {
	return shiftPtr(hPtr, ghostWidth, Tsize);
}

void *PlacesPartition::getRightBuffer() {
	void *retVal = shiftPtr(hPtr, numElements, Tsize);
	if (0 == rank) {
		// there is no left ghost width, shift a negative direction
		retVal = shiftPtr(retVal, -1 * ghostWidth, Tsize);
	}
	return retVal;
}

void *PlacesPartition::getLeftGhost() {
	return hPtr; // this is where hPtr starts
}

void *PlacesPartition::getRightGhost() {
	void *retVal = shiftPtr(hPtr, numElements, Tsize);
	if (rank > 0) {
		// we started at -ghostWidth elements.
		retVal = shiftPtr(retVal, ghostWidth, Tsize);
	}
	return retVal;
}

dim3 PlacesPartition::blockDim() {
	return dims[0];
}

dim3 PlacesPartition::threadDim() {
	return dims[1];
}

void PlacesPartition::setIdealDims() {
	int numBlocks = (numElements - 1) / THREADS_PER_BLOCK + 1;
	dim3 blockDim(numBlocks, 1, 1);

	int nThr = (numElements - 1) / numBlocks + 1;
	dim3 threadDim(nThr, 1, 1);

	dims[0] = blockDim;
	dims[1] = threadDim;
}

int PlacesPartition::getPlaceBytes() {
	return Tsize;
}

} /* namespace mass */
