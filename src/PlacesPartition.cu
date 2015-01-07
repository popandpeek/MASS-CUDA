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
//#include "DllClass.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Logger.h"

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
	Logger::debug("Entering PlacesPartition constructor.");
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
	int numRanks = 1;//Mass::getPlaces(handle)->getNumPartitions();
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
	Logger::debug("Setting ghost width in partition.");
	// start at 1 because we never want to factor in x step
	for (int i = 1; i < n; ++i) {
		ghostWidth += dimensions[i];
	}

	// set pointer
//	Places *places = 1;//Mass::getPlaces(handle);
//	if(NULL == places){
//		Logger::debug("Places not found under this handle.");
//	}
//	if (0 == rank) {
//		Logger::debug("Setting rank 0's hPtr.");
//		hPtr = places->dllClass->placeElements;
//	} else {
//		Logger::debug("Setting rank %d's hPtr.", getRank());
//		hPtr = shiftPtr(places->dllClass->placeElements, rank * numElements - ghostWidth, Tsize);
//	}
	Logger::debug("Done setting ghost width in partition.");
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
	Logger::debug("Setting ideal dims.");
	int numBlocks = (numElements - 1) / THREADS_PER_BLOCK + 1;
	Logger::debug("Creating block dim.");
	dim3 blockDim(numBlocks, 1, 1);

	int nThr = (numElements - 1) / numBlocks + 1;
	Logger::debug("Creating thread dim.");
	dim3 threadDim(nThr, 1, 1);


	Logger::debug("Assigning dims.");
	dims[0] = blockDim;
	dims[1] = threadDim;
	Logger::debug("Done setting ideal dims.");
}

int PlacesPartition::getPlaceBytes() {
	return Tsize;
}

} /* namespace mass */
