/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Mass.h"
#include "Place.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Dispatcher.h"
#include "MassException.h"

using namespace std;

namespace mass {

Places::~Places() {
	if (NULL != elements) {
		free(elements);
	}
	delete[] elemPtrs;
	partitions.empty();
}

int Places::getDimensions() {
	return numDims;
}

int *Places::size() {
	return dimensions;
}

int Places::getHandle() {
	return handle;
}

void Places::callAll(int functionId) {
	callAll(functionId, NULL, 0);
}

void Places::callAll(int functionId, void *argument, int argSize) {
	dispatcher->callAllPlaces(this, functionId, argument, argSize);
}

void *Places::callAll(int functionId, void *arguments[], int argSize,
		int retSize) {
	return dispatcher->callAllPlaces(this, functionId, arguments, argSize,
			retSize);
}

void Places::exchangeAll(int functionId, std::vector<int*> *destinations) {
	dispatcher->exchangeAllPlaces(this, functionId, destinations);
}

void Places::exchangeBoundary() {
	dispatcher->exchangeBoundaryPlaces(this);
}

Place** Places::getElements() {
	dispatcher->refreshPlaces(this);
	return elemPtrs;
}

int Places::getNumPartitions() {
	return partitions.size();
}

void Places::setPartitions(int numParts) {

	// make sure update is necessary
	if (!partitions.size() == numParts && numParts > 0) {
		dispatcher->refreshPlaces(this); // get current data

		char *copyStart = (char*) elements; // this is a hack to allow arithmatic on a void* pointer
		int numRanks = partitions.size();

		int sliceSize = numElements / numParts;
		int remainder = numElements % numParts;

		PlacesPartition *part = new PlacesPartition( handle, 0, sliceSize,
				boundary_width, numDims, dimensions, Tsize );

		for (int i = 0; i < numRanks; ++i) {

		}
	}
}

void Places::setTsize(int size) {
	Tsize = size;
}

int Places::getTsize() {
	return Tsize;
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

vector<int> Places::getIndexVector(int rowMajorIdx) {
	vector<int> indices; // return value

	for (int i = numDims - 1; i > 0; --i) {
		int idx = rowMajorIdx % dimensions[i];
		rowMajorIdx /= dimensions[i];
		indices.insert(indices.begin(), idx);
	}

	return indices;
}

PlacesPartition *Places::getPartition(int rank) {
	if (rank < 0 || rank >= partitions.size()) {
		throw MassException(
				"Out of bounds rank specified in Places::getPartition()");
	}
	return partitions[rank];
}

Places::Places(int handle, std::string className, void *argument, int argSize, int dimensions,
		int size[], int boundary_width) {
	this->handle = handle;
	this->numDims = dimensions;
	this->dimensions = size;
	this->boundary_width = boundary_width;
	this->argument = argument;
	this->argSize = argSize;
	this->dispatcher = Mass::dispatcher; // the GPU dispatcher

	this->numElements = 1;
	for (int i = 0; i < numDims; ++i) {
		numElements *= size[i];
	}
	this->elemPtrs = new Place*[numElements];
	this->Tsize = 0;
	this->elements = NULL;
}

} /* namespace mass */
