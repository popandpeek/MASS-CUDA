

#include <stdarg.h>

#include "Places.h"
#include "Place.h"
#include "Dispatcher.h"
#include "Logger.h"

using namespace std;

namespace mass {

Places::~Places() {
	for (int i = 0; i < elemPtrs.size(); ++i) {
		delete[] elemPtrs.at(i);
	}

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


int Places::getPlacesStride() {
	placesStride = dispatcher->getPlacesStride(handle);
	return placesStride;
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

void Places::exchangeAll(std::vector<int*> *destinations) {
	dispatcher->exchangeAllPlaces(handle, destinations);
}

/**
 *  This function causes all Place elements to call the function specified on all neighboring
 *  place elements. In addition to the fuctionality of the exchangeAllPlaces function specified above 
 *  it also takes functionId as a parameter and arguments to that functiom. 
 *  When the data is collected from the neighboring places, 
 *  the specified function is executed on all of the places with specified parameters.
 *  The rationale behind implemening this version of exchangeAllPlaces is performance optimization:
 *  the data cached during data collection step can be used for the data calculation and thus minimize
 *  the number of memeory fetches and improve performance.
 */
void Places::exchangeAll(std::vector<int*> *destinations, int functionId, void *argument, int argSize) {
	dispatcher->exchangeAllPlaces(handle, destinations, functionId, argument, argSize);
}

Place** Places::getElements() {
	std::vector<Place**> elemPtrsVec = dispatcher->refreshPlaces(handle);
	mass::Place** retVals = new Place*[numElements];
	int placesStride = this->getPlacesStride();
	Logger::debug("Places::getElements(): placesStride = %d; total places = %d; placesVec.size() = %d", 
			placesStride, numElements, elemPtrsVec.size());
	int count = 0;

	for (int i = 0; i < elemPtrsVec.size(); ++i) {
		int j = 0;
		for (; j < placesStride; ++j) {
			retVals[count] = elemPtrsVec.at(i)[j];
			// Logger::debug("retVals[count] location is %d; tmp_ptr[j] location is == %d", count, j);
			count++;
		}
	}

	elemPtrs = elemPtrsVec;
	Logger::debug("Places::getElements: Finished.");
	return retVals;
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

int Places::getRowMajorIdx(...) {
	int *indices = new int[numDims];
	va_list args;
	va_start(args, numDims);
	for ( int i = 0; i < numDims; i++ ){ // Loop until all numbers are added
	    indices[i] = va_arg ( args, int );
	}
	va_end(args);
	int rmi = getRowMajorIdx(indices);
	delete[] indices;
	return rmi;
}

vector<int> Places::getIndexVector(int rowMajorIdx) {
	vector<int> indices; // return value

	for (int i = numDims - 1; i >= 0; --i) {
		int idx = rowMajorIdx % dimensions[i];
		rowMajorIdx /= dimensions[i];
		indices.insert(indices.begin(), idx);
	}

	return indices;
}

Places::Places(int handle, int dimensions, int size[], Dispatcher *d) {
	Logger::debug("Inside Places private constructor.");
	this->handle = handle;
	this->dispatcher = d;
	this->dimensions = size;
	this->numDims = dimensions;
	this->elemPtrs = {};
	this->numElements = 1;
	for (int i = 0; i < dimensions; ++i) {
		this->numElements *= size[i];
	}
}

} /* namespace mass */
