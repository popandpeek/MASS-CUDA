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
#include "Logger.h"

namespace mass {

inline void *shiftPtr(void *origin, int qty, int Tsize) {
	char *retVal = (char*) origin;
	retVal += Tsize * qty;
	return retVal;
}

PlacesPartition::PlacesPartition(int handle, int rank, int numElements,
		int ghostWidth, int n, int *dimensions) {
	Logger::print("Inside PlacesPartition constructor.");
	this->hPtr = NULL;
	this->handle = handle;
	this->rank = rank;
	this->numElements = numElements;
	Logger::print("copied all ards into class members");
	setGhostWidth(ghostWidth, n, dimensions);
	Logger::print("after setGhostWidth");
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
int PlacesPartition::sizeWithGhosts() {
	int numRanks = 1; //Mass::getPlaces(handle)->getNumPartitions();
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
 *  Returns the handle associated with this PlacesPartition object that was set at construction.
 */
int PlacesPartition::getHandle() {
	return handle;
}

void PlacesPartition::setGhostWidth(int width, int n, int *dimensions) {
	ghostWidth = width;
	Logger::debug("Setting ghost width in partition.");
	// start at 1 because we never want to factor in x step
	for (int i = 1; i < n; ++i) {
		ghostWidth *= dimensions[i];
	}

	// prevent indexing out of bounds on thin sections with wide ghosts
	if (ghostWidth > numElements) {
		ghostWidth = numElements;
	}

	Logger::debug("Done setting ghost width in partition.");
}

Place *PlacesPartition::getLeftBuffer() {
	return hPtr[ghostWidth];
}

Place *PlacesPartition::getRightBuffer() {
	if (0 == rank) {
		// there is no left ghost width, shift a negative direction
		return hPtr[numElements - ghostWidth];
	}
	return hPtr[numElements];
}

Place *PlacesPartition::getLeftGhost() {
	return hPtr[0]; // this is where hPtr starts
}


dim3 PlacesPartition::blockDim() {
	return dims[0];
}

dim3 PlacesPartition::threadDim() {
	return dims[1];
}

void PlacesPartition::setSection(Place** start) {
	hPtr = start;
}

void PlacesPartition::setIdealDims() {
	Logger::debug("Setting ideal dims.");
	int numBlocks = (numElements - 1) / THREADS_PER_BLOCK + 1;
	Logger::debug("Creating block dim.");
	dim3 blockDim(numBlocks);

	int nThr = (numElements - 1) / numBlocks + 1;
	Logger::debug("Creating thread dim.");
	dim3 threadDim(nThr);

	Logger::debug("Assigning dims.");
	dims[0] = blockDim;
	dims[1] = threadDim;
	Logger::debug("Done setting ideal dims.");
}

int PlacesPartition::getPlaceBytes() {
	return Tsize;
}

} /* namespace mass */
