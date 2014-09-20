/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Mass.h"
#include "Places.h"
#include "Model.h"
#include "Dispatcher.h"

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
	dispatcher->refreshPlaces(handle);
	return elements;
}

int Places::getHandle() {
	return handle;
}

void Places::callAll(int functionId) {
	callAll(functionId, NULL, 0);
}

void Places::callAll(int functionId, void *argument, int argSize) {
	dispatcher->callAllPlaces(handle, functionId, argument, argSize);
}

void *Places::callAll(int functionId, void *arguments[], int argSize,
		int retSize) {
	return dispatcher->callAllPlaces(handle, functionId, arguments, argSize, retSize);
}

void Places::exchangeAll(int functionId, std::vector<int*> *destinations) {
	dispatcher->exchangeAllPlaces(handle, functionId, destinations);
}

void Places::exchangeBoundary() {
	dispatcher->exchangeBoundaryPlaces(handle);
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
