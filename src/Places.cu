/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Places.h"
#include "Place.h"
#include "Dispatcher.h"
#include "Logger.h"

using namespace std;

namespace mass {

Places::~Places() {
	delete[] elemPtrs;
	delete[] dimensions;
}

int Places::getDimensions() {
	return numDims;
}

int *Places::size() {
	return dimensions;
}

int Places::getNumPlaces() {
	return numElements;
}

int Places::getHandle() {
	return handle;
}

void Places::callAll(int functionId) {
	Logger::debug("Entering callAll(int functionId)");
	callAll(functionId, NULL, 0);
}

void Places::callAll(int functionId, void *argument, int argSize) {
	Logger::debug(
			"Entering callAll(int functionId, void *argument, int argSize)");
	dispatcher->callAllPlaces(handle, functionId, argument, argSize);
}

void *Places::callAll(int functionId, void *arguments[], int argSize,
		int retSize) {
	return dispatcher->callAllPlaces(handle, functionId, arguments, argSize,
			retSize);
}

void Places::exchangeAll(int functionId, std::vector<int*> *destinations) {
	dispatcher->exchangeAllPlaces(handle, functionId, destinations);
}

void Places::exchangeBoundary() {
	dispatcher->exchangeBoundaryPlaces(handle);
}

Place** Places::getElements() {
	// TODO can I avoid refresh every time?
	elemPtrs = dispatcher->refreshPlaces(handle);
	return elemPtrs;
}

int Places::getRowMajorIdx(int *indices) {
	// a single X will pass over y*z elements,
	// a single Y will pass over z elements, and a Z will pass over 1 element.
	// each dimension will be removed from numElements before calculating the
	// size of each index's "step"
	int multiplier = (int) numElements;
	int rmi = 0; // accumulater for row major index

	for (int i = 0; i < numDims; i++) {
		multiplier /= dimensions[i]; // remove dimension from multiplier
		int idx = indices[i]; // get an index and check validity
		if (idx < 0 || idx >= dimensions[i]) {
			throw MassException("The indices provided are out of bounds");
		}
		rmi += multiplier * idx; // calculate step
	}

	return rmi;
}

int Places::getRowMajorIdx(vector<int> indices) {
	return getRowMajorIdx(&indices[0]);
}

vector<int> Places::getIndexVector(int rowMajorIdx) {
	vector<int> indices; // return value

	for (int i = numDims - 1; i > 0; --i) {
		int idx = rowMajorIdx % dimensions[i];
		rowMajorIdx /= dimensions[i];
		indices.insert(indices.begin(), idx);
	}

	return indices;
}

Places::Places(int handle, int dimensions, int size[], Dispatcher *d) {
	this->handle = handle;
	this->dispatcher = d;
	this->dimensions = size;
	this->numDims = dimensions;
	this->elemPtrs = NULL;
	this->numElements = 1;
	for (int i = 0; i < dimensions; ++i) {
		this->numElements *= size[i];
	}
}

} /* namespace mass */
