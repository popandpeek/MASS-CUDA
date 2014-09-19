/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Places.h"

namespace mass {

Places::~Places() {
	if (NULL != elements) {
		delete[] elements;
	}
}

int Places::getDimensions() {
	return numDims;
}

int *Places::size() {
	return dimensions;
}

Place* Places::getElements() {
	return elements;
}

int Places::getHandle() {
	return handle;
}

void Places::callAll(int functionId) {
	//TODO send call all command to dispatcher
}

void Places::callAll(int functionId, void *argument, int argSize) {
	//TODO send call all command to dispatcher
}

void *Places::callAll(int functionId, void *arguments[], int argSize,
		int retSize) {
	//TODO send call all command to dispatcher
	return NULL;
}

void Places::exchangeAll(int handle, int functionId,
		std::vector<int*> *destinations) {
	//TODO send exchange all command to dispatcher
}

void Places::exchangeBoundary() {
	//TODO send cexchange boundary command to dispatcher
}

Places::Places(int handle, int boundary_width, void *argument, int argSize,
		int dimensions, int size[]) {
	this->handle = handle;
	this->elements = elements;
	this->numDims = dimensions;
	this->dimensions = size;
	this->boundary_width = boundary_width;
	this->numElements = 1;
	for (int i = 0; i < numDims; ++i) {
		numElements *= this->dimensions[i];
	}
	elements = NULL;
}

} /* namespace mass */
