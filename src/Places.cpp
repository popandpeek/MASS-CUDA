/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include <sstream> // stringstream
#include "DllClass.h"
#include "Mass.h"
#include "Place.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Dispatcher.h"
#include "MassException.h"

using namespace std;

namespace mass {

Places::~Places() {
	delete[] elemPtrs;
	delete[] dimensions;
	// destroy dll class
	delete dllClass;
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
	Mass::log("Places::setPartitions(int numParts) is commented out.");
/*
	// make sure update is necessary
	if (partitions.size() != numParts && numParts > 0) {
		stringstream ss;
		ss << "Setting places " << handle << " partitions to " << numParts;
		Mass::log(ss.str());

		dispatcher->refreshPlaces(this); // get current data

		// remove old partitions
		map<int, PlacesPartition*>::iterator it = partitions.begin();
		while (it != partitions.end()) {
			delete it->second;
			++it;
		}

		char *copyStart = (char*) dllClass->placeElements; // this is a hack to allow arithmetic on a void* pointer

		int sliceSize = numElements / numParts;
		int remainder = numElements % numParts;

		// there is always at least one partition
		PlacesPartition *part = new PlacesPartition(handle, 0, sliceSize,
				boundary_width, numDims, dimensions, Tsize);
		part->hPtr = copyStart;
		copyStart += Tsize * sliceSize - part->ghostWidth; // subtract ghost width as rank 0 has none
		partitions[part->getRank()] = part;

		for (int i = 1; i < numParts; ++i) {
			// last rank will have remainder elements, not sliceSize
			int sz = (numParts - 1 == i) ? remainder : sliceSize;
			// set hPtr
			part = new PlacesPartition(handle, i, sz, boundary_width,
					numDims, dimensions, Tsize);
			part->hPtr = copyStart;
			copyStart += Tsize * sliceSize;
			partitions[part->getRank()] = part;
		}
	}
*/
	// TODO set corresponding agents partitions
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
		stringstream ss;
		ss << "Requested partition " << rank << " but there are only 0 - "
				<< getNumPartitions() - 1 << " are valid.";
		Mass::log(ss.str());
		throw MassException(ss.str());
	}
	return partitions[rank];
}

Places::Places(int handle, std::string className, void *argument, int argSize,
		int dimensions, int size[], int boundary_width) {
	this->handle = handle;
	this->numDims = dimensions;

	this->numElements = 1;
	this->dimensions = new int[numDims];
	for (int i = 0; i < numDims; ++i) {
		this->dimensions[i] = size[i];
		numElements *= size[i];
	}

	this->boundary_width = boundary_width;
//	this->argument = argument;
//	this->argSize = argSize;
	this->dispatcher = Mass::dispatcher; // the GPU dispatcher
	this->elemPtrs = new Place*[numElements];
	this->Tsize = 0;
	this->classname = className;
	init_all(argument, argSize);
	dispatcher->configurePlaces(this);
}

void Places::init_all(void *argument, int argSize) {
	// For debugging
	stringstream ss;

	ss << "Places Initialization:\n" << "\thandle = " << handle << "\n\tclass = "
			<< classname << "\n\targument_size = " << argSize
			<< "\n\targument = " << *((char *) argument) << "\n\tdimensionality = "
			<< numDims << "\n\tdimensions = { " << dimensions[0];
	for (int i = 1; i < numDims; i++) {
		ss << " ," << dimensions[i];
	}
	ss << " }";

	// Print the current working directory
	char buf[200];
	getcwd(buf, 200);
	ss << "\n\tCurrent working directory: " << buf;
	Mass::log(ss.str());

//	Mass::log("Place initialization is commented out.");

	// load the construtor and destructor
	dllClass = new DllClass(classname);

	// instanitate a new place
	Mass::log("Attempting to instantiate a place.");
	Place *protoPlace = (Place *) (dllClass->instantiate(argument));
	Mass::log("Place instantiation successful.");
	this->Tsize = protoPlace->placeSize();

	// set common place fields
	protoPlace->numDims = numDims;
	for (int i = 0; i < numDims; ++i) {
		protoPlace->size[i] = dimensions[i];
	}

	//  space for an entire set of place instances
	dllClass->placeElements = malloc(numElements * Tsize);

	// char is used to allow void* arithmatic in bytes
	char *copyDest = (char*) dllClass->placeElements;

	for (int i = 0; i < numElements; ++i) {
		// memcpy protoplace
		memcpy(copyDest, protoPlace, Tsize);
		((Place *) copyDest)->index = i; // set the unique index
		copyDest += Tsize; // increment copy destination
	}

	dllClass->destroy(protoPlace); // we no longer need this

}

} /* namespace mass */
