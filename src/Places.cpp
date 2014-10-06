/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Mass.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Dispatcher.h"

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

void Places::addPartitions(std::vector<PlacesPartition*> parts) {

	// make sure add is valid
	if (NULL == elements) {
		elements = malloc(Tsize * numElements);
	}

	char *copyStart = (char*) elements; // this is a hack to allow arithmatic on a void* pointer
	int numRanks = parts.size();

	for (int i = 0; i < numRanks; ++i) {
		PlacesPartition* part = parts[i];
		partitions[i] = part;
	}
}

void Places::setTsize(int size) {
	Tsize = size;
}

int Places::getTsize() {
	return Tsize;
}

int Places::getRowMajorIdx(...) {
	// a single X will pass over y*z elements,
	// a y will pass over z elements, and a z will pass over 1 element.
	// each dimension will be removed from numElements before calculating the
	// size of each index's "step"
	int multiplier = (int) numElements;
	int rmi = 0; // accumulater for row major index

	a_list ap;
	va_start(ap, numDims);
	for (int i = 0; i < numDims; i++) {
		multiplier /= dimensions[i]; // remove dimension from multiplier
		int idx = va_arg(ap, int); // get an index and check validity
		if (idx < 0 || idx >= dimensions[i]) {
			throw MassException("The indices provided are out of bounds");
		}
		rmi += multiplier * idx; // calculate step
	}
	va_end(ap);

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

Places::Places(int handle, int boundary_width, void *argument, int argSize,
		int dimensions, int size[]) {
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
