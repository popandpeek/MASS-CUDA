/**
 *  @file Places.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <cuda_runtime.h>
#include <sstream> // stringstream
#include "DllClass.h"
#include "Mass.h"
#include "Place.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Dispatcher.h"
#include "MassException.h"
#include "cudaUtil.h"

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

int Places::getNumPlaces() {
	return numElements;
}

int Places::getHandle() {
	return handle;
}

void Places::callAll(int functionId) {
	Mass::logger.debug("Entering callAll(int functionId)");
	callAll(functionId, NULL, 0);
}

void Places::callAll(int functionId, void *argument, int argSize) {
	Mass::logger.debug(
			"Entering callAll(int functionId, void *argument, int argSize)");
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
	Mass::logger.debug("Entering Places::setPartitions(int numParts).");

	// make sure update is necessary
	if (partitions.size() != numParts && numParts > 0) {
		Mass::logger.debug("Setting places %d partitions to %d.", handle,
				numParts);

		if (partitions.size() > 0) { // this isn't the initial setPartitions call
			Mass::logger.debug("Refreshing data and removing old partitions.");
			dispatcher->refreshPlaces(this); // get current data

			// remove old partitions
			map<int, PlacesPartition*>::iterator it = partitions.begin();
			while (it != partitions.end()) {
				map<int, PlacesPartition*>::iterator tmp = it;
				it++;
				delete tmp->second;
				tmp->second = NULL;
				partitions.erase(tmp);
				++it;
			}
		}

		char *copyStart = (char*) dllClass->placeElements; // this is a hack to allow arithmetic on a void* pointer

		int sliceSize = numElements / numParts;
		int remainder = numElements % numParts;

		Mass::logger.debug(
				"Partitions info:\n\tnumElements = %d\n\tsliceSize = %d\n\tremainder = %d",
				numElements, sliceSize, remainder);

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
			Mass::logger.debug("Adding partition %d\n\tsize = %d", i, sz);
			part = new PlacesPartition(handle, i, sz, boundary_width, numDims,
					dimensions, Tsize);
			part->hPtr = copyStart;
			copyStart += Tsize * sliceSize;
			partitions[part->getRank()] = part;
		}
	} else {
		Mass::logger.debug(
				"Number of partitions specified is either invalid or does not change the partition count.");
	}
	Mass::logger.debug("Done partitioning places");
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

		Mass::logger.debug(
				"Requested partition %d but there are only 0 - %d are valid.",
				rank, getNumPartitions() - 1);
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
	this->dispatcher = Mass::dispatcher; // the GPU dispatcher
	this->elemPtrs = new Place*[numElements];
//	memset(elemPtrs,0,numElements * sizeof(Place*));
	this->Tsize = 0;
	this->classname = className;
	init_all(argument, argSize);
	Mass::placesMap[handle] = this;
	dispatcher->configurePlaces(this);
}

void Places::init_all(void *argument, int argSize) {
	// For debugging
	stringstream ss;

	ss << "Places Initialization:\n" << "\thandle = " << handle
			<< "\n\tclass = " << classname << "\n\targument_size = " << argSize
			<< "\n\targument = " << *((char *) argument)
			<< "\n\tdimensionality = " << numDims << "\n\tdimensions = { "
			<< dimensions[0];
	for (int i = 1; i < numDims; i++) {
		ss << " ," << dimensions[i];
	}
	ss << " }";

	// Print the current working directory
	char buf[200];
	getcwd(buf, 200);
	ss << "\n\tCurrent working directory: " << buf;
	char cstr[500];
	strcpy(cstr, ss.str().c_str());
	Mass::logger.debug(cstr);

	// load the construtor and destructor
	dllClass = new DllClass(classname);

	// instanitate a new place
	Place *protoPlace = (Place *) dllClass->instantiate(argument);
	this->Tsize = protoPlace->placeSize();

	// set common place fields
	protoPlace->numDims = numDims;
	for (int i = 0; i < numDims; ++i) {
		protoPlace->size[i] = dimensions[i];
	}
	if (numDims < MAX_DIMS) {
		// set remaining dimensions to zero
		memset(protoPlace->size + numDims, 0,
				sizeof(int) * (MAX_DIMS - numDims));
	}

	//  space for an entire set of place instances
	void *elems;
	Mass::logger.debug("cudaMallocHost call");
	CATCH(cudaMallocHost((void**) elems, numElements * Tsize));
	Mass::logger.debug("cudaMallocHost call finished");
	dllClass->placeElements = elems;

	// char is used to allow void* arithmatic in bytes
	char *copyDest = (char*) dllClass->placeElements;

	Mass::logger.debug("Copying protoplace to each element.");
	for (int i = 0; i < numElements; ++i) {
		// memcpy protoplace
		memcpy((void*) copyDest, protoPlace, Tsize);
		elemPtrs[i] = (Place *) copyDest;
		elemPtrs[i]->index = i; // set the unique index
		copyDest += Tsize; // increment copy destination
	}

	dllClass->destroy(protoPlace); // we no longer need this
	Mass::logger.debug("Exiting Places::init_all");
}


Places::Places(int handle, Place *proto, void *argument, int argSize,
			int dimensions, int size[], int boundary_width){
  this->handle = handle;
	this->numDims = dimensions;

	this->numElements = 1;
	this->dimensions = new int[numDims];
	for (int i = 0; i < numDims; ++i) {
		this->dimensions[i] = size[i];
		numElements *= size[i];
	}

	this->boundary_width = boundary_width;
	this->dispatcher = Mass::dispatcher; // the GPU dispatcher
	this->elemPtrs = new Place*[numElements];
//	memset(elemPtrs,0,numElements * sizeof(Place*));
	this->Tsize = proto->placeSize();
	this->classname = "";
	init_all(proto, argument, argSize);
	Mass::placesMap[handle] = this;
	dispatcher->configurePlaces(this);
}

void Places::init_all(Place * proto, void *argument, int argSize) {
	// For debugging
	stringstream ss;

	ss << "Places Initialization:\n" << "\thandle = " << handle
			<< "\n\targument_size = " << argSize
			<< "\n\targument = " << *((char *) argument)
			<< "\n\tdimensionality = " << numDims << "\n\tdimensions = { "
			<< dimensions[0];
	for (int i = 1; i < numDims; i++) {
		ss << " ," << dimensions[i];
	}
	ss << " }";

	// Print the current working directory
	char buf[200];
	getcwd(buf, 200);
	ss << "\n\tCurrent working directory: " << buf;
	char cstr[500];
	strcpy(cstr, ss.str().c_str());
	Mass::logger.debug(cstr);
  dllClass = new DllClass(proto);
	this->Tsize = proto->placeSize();

	// set common place fields
	proto->numDims = numDims;
	for (int i = 0; i < numDims; ++i) {
		proto->size[i] = dimensions[i];
	}
  
	if (numDims < MAX_DIMS) {
		// set remaining dimensions to zero
		memset(proto->size + numDims, 0,
				sizeof(int) * (MAX_DIMS - numDims));
	}

	//  space for an entire set of place instances
	void *elems;
	Mass::logger.debug("cudaMallocHost call");
	CATCH(cudaMallocHost((void**) elems, numElements * Tsize));
	Mass::logger.debug("cudaMallocHost call finished");
	dllClass->placeElements = elems;

	// char is used to allow void* arithmatic in bytes
	char *copyDest = (char*) dllClass->placeElements;

	Mass::logger.debug("Copying protoplace to each element.");
	for (int i = 0; i < numElements; ++i) {
		// memcpy protoplace
		memcpy((void*) copyDest, proto, Tsize);
		elemPtrs[i] = (Place *) copyDest;
		elemPtrs[i]->index = i; // set the unique index
		copyDest += Tsize; // increment copy destination
	}

	Mass::logger.debug("Exiting Places::init_all");
}

} /* namespace mass */
