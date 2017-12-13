/**
 *  @file Dispatcher.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <sstream>
#include <algorithm>  // array compare
#include <iterator>

#include "Dispatcher.h"
#include "cudaUtil.h"
#include "Logger.h"

#include "DeviceConfig.h"
#include "Place.h"
#include "PlacesPartition.h"
#include "Places.h"
#include "DataModel.h"

using namespace std;

namespace mass {

__global__ void callAllPlacesKernel(Place **ptrs, int nptrs, int functionId,
		void *argPtr) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < nptrs) {
		ptrs[idx]->callMethod(functionId, argPtr);
	}
}

/**
 * neighbors is converted into a 1D offset of relative indexes before calling this function
 */
 __global__ void setNeighborPlacesKernel(Place **ptrs, int nptrs, GlobalConsts *glob) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < nptrs) {
		int nNeighbors = glob->nNeighbors;

		__shared__ int offsets[4];
		memcpy(offsets,glob->offsets,sizeof(int) * nNeighbors);

		PlaceState *state = ptrs[idx]->getState();
		int nSkipped = 0;
		for (int i = 0; i < nNeighbors; ++i) {
			int j = idx + offsets[i];
			if (j >= 0 && j < nptrs) {
				state->neighbors[i - nSkipped] = ptrs[j];
				state->inMessages[i - nSkipped] = ptrs[j]->getMessage();
			} else {
				nSkipped++;
			}
		}
	}
}

Dispatcher::Dispatcher() {
	model = NULL;
	initialized = false;
	neighborhood = NULL;
}

struct DeviceAndMajor {
	DeviceAndMajor(int device, int major) {
		this->device = device;
		this->major = major;
	}
	int device;
	int major;
};
bool compFunction (DeviceAndMajor i,DeviceAndMajor j) { return (i.major>j.major); }

void Dispatcher::init() {
	if (!initialized) {
		initialized = true;
		Logger::debug(("Initializing Dispatcher"));
		int gpuCount;
		cudaGetDeviceCount(&gpuCount);

		if (gpuCount == 0) {
			throw MassException("No GPU devices were found.");
		}

		vector<DeviceAndMajor> devices;
		for (int d = 0; d < gpuCount; d++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, d);

			Logger::debug("Device %d has compute capability %d.%d", d,
					deviceProp.major, deviceProp.minor);

			DeviceAndMajor deviceAndMajor = DeviceAndMajor(d, deviceProp.major);
			devices.push_back(deviceAndMajor);
		}

		//Sort devices by compute capability in descending order:
		std::sort (devices.begin(), devices.end(), compFunction);

		// Pick the device with the highest compute capability for simulation:
		deviceInfo = new DeviceConfig(devices[0].device);
		model = new DataModel();
	}
}

Dispatcher::~Dispatcher() {
	Logger::debug("Freeing deviceConfig");
	deviceInfo -> freeDevice();
}

Place** Dispatcher::refreshPlaces(int handle) {
	if (initialized) {
		Logger::debug("Entering Dispatcher::refreshPlaces");

		int stateSize = model->getPlacesModel(handle)->getStateSize();

		PlacesPartition* p = partInfo->getPlacesPartition(handle);
		void *devPtr = deviceInfo->getPlaceState(handle); // gets the state belonging to this partition
		int qty = p->sizeWithGhosts();
		int bytes = stateSize * qty;
		CATCH(cudaMemcpy(p->getLeftBuffer()->getState(), devPtr, bytes, D2H));

		Logger::debug("Exiting Dispatcher::refreshPlaces");
	}

	return model->getPlacesModel(handle)->getPlaceElements();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument,
		int argSize) {
	if (initialized) {
		Logger::debug("Calling all on places[%d]", placeHandle);

		Partition* partition = model->getPartition();

		if (partInfo == NULL) { // the partition needs to be loaded
			deviceInfo->loadPartition(partition, placeHandle);
			partInfo = partition;

			Logger::debug("Loaded partition[%d]", placeHandle);
		} 

		// load any necessary arguments
		void *argPtr = NULL;
		if (argument != NULL) {
			deviceInfo->load(argPtr, argument, argSize);
			Logger::debug("Loaded device\n");
		}

		Logger::debug("Calling callAllPlacesKernel");
		PlacesPartition *pPart = partition->getPlacesPartition(placeHandle);
		callAllPlacesKernel<<<pPart->blockDim(), pPart->threadDim()>>>(
				deviceInfo->getDevPlaces(placeHandle), pPart->sizeWithGhosts(),
				functionId, argPtr);
		CHECK();

		if (argPtr != NULL) {
			Logger::debug("Freeing device args.");
			cudaFree(argPtr);
		}

		Logger::debug("Exiting Dispatcher::callAllPlaces()");
	}
}

void *Dispatcher::callAllPlaces(int handle, int functionId, void *arguments[],
		int argSize, int retSize) {
	// perform call all
	callAllPlaces(handle, functionId, arguments, argSize);
	// get data from GPUs
	refreshPlaces(handle);
	// get necessary pointers and counts
	int qty = model->getPlacesModel(handle)->getNumElements();
	Place** places = model->getPlacesModel(handle)->getPlaceElements();
	void *retVal = malloc(qty * retSize);
	char *dest = (char*) retVal;

	for (int i = 0; i < qty; ++i) {
		// copy messages to a return array
		memcpy(dest, places[i]->getMessage(), retSize);
		dest += retSize;
	}
	return retVal;
}

bool compArr(int* a, int aLen, int *b, int bLen) {
	if (aLen != bLen) {
		return false;
	}

	for (int i = 0; i < aLen; ++i) {
		if (a[i] != b[i])
			return false;
	}
	return true;
}

bool Dispatcher::updateNeighborhood(int handle, vector<int*> *vec) {
	if (vec == neighborhood) { //no need to update
		return false;
	}

	neighborhood = vec;
	int nNeighbors = vec->size();

	int *offsets = new int[nNeighbors];
	PlacesModel *p = model->getPlacesModel(handle);
	int nDims = p->getNumDims();
	int *dimensions = p->getDims();
	int numElements = p->getNumElements();

	// calculate an offset for each neighbor in vec
	for (int j = 0; j < vec->size(); ++j) {
		int *indices = (*vec)[j];
		int offset = 0; // accumulater for row major offset
		int multiplier = 1;

		// a single X will pass over y*z elements,
		// a single Y will pass over z elements, and a Z will pass over 1 element.
		// each dimension will be removed from multiplier before calculating the
		// size of each index's "step"
		for (int i = 0; i < nDims; i++) {
			// convert from raster to cartesian coordinates
			if (1 == i) {
				offset -= multiplier * indices[i];
			} else {
				offset += multiplier * indices[i];
			}

			multiplier *= dimensions[i]; // remove dimension from multiplier
		}
		offsets[j] = offset;
	}

	GlobalConsts c = deviceInfo->getGlobalConstants();
	memcpy(c.offsets, offsets, nNeighbors * sizeof(int));
	c.nNeighbors = nNeighbors;
	deviceInfo->updateConstants(c);

	delete [] offsets;
	return true;
}

void Dispatcher::exchangeAllPlaces(int handle, std::vector<int*> *destinations) {

	updateNeighborhood(handle, destinations);

	Place** ptrs = deviceInfo->getDevPlaces(handle);
	int nptrs = deviceInfo->countDevPlaces(handle);
	PlacesPartition *p = model->getPartition()->getPlacesPartition(handle);

	setNeighborPlacesKernel<<<p->blockDim(), p->threadDim()>>>(ptrs, nptrs,
			deviceInfo->d_glob);
	CHECK();
}

void Dispatcher::unloadDevice(DeviceConfig *device) {
	Logger::print("Inside Dispatcher::unloadDevice\n");
	if (partInfo != NULL) {
		Logger::print("device != NULL\n");
		Partition* p = partInfo;
		map<int, PlacesPartition*> places = p->getPlacesPartitions();  //place partitions by handle

		Logger::print("p->getPlacesPartitions() finished\n");
		map<int, PlacesPartition*>::iterator itP = places.begin();
		while (itP != places.end()) {
			refreshPlaces(itP->first);
			Logger::print("refreshed places\n");
		}

		deviceInfo = NULL;
		partInfo = NULL;
	}
}

}// namespace mass

