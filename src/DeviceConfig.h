/**
 *  @file DeviceConfig.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#pragma once

#include <map>
#include "cudaUtil.h"
#include "Logger.h"
#include "PlaceState.h"
#include "Partition.h"

namespace mass {

// forward declarations
class Agent;
class Place;

struct PlaceArray {
	Place** devPtr;
	void *devState;
	int qty;
};

struct AgentArray {
	Agent** devPtr;
	void *devState;
	int qty;
};

/**
 *  This class represents a computational resource. In most cases it will
 *  represent a GPU, but it could also be used to encapsulate a CPU
 *  computing resource.
 */
class DeviceConfig {
	friend class Dispatcher;

public:
	DeviceConfig();
	DeviceConfig(int device);

	virtual ~DeviceConfig();
	void freeDevice();

	void setAsActiveDevice();

	void load(void*& destination, const void* source, size_t bytes);
	void unload(void* destination, void* source, size_t bytes);

	void loadPartition(Partition* partition, int placeHandle);

	/*
	 * Place Mutators
	 */

	Place** getDevPlaces(int handle);
	void* getPlaceState(int handle);
	int countDevPlaces(int handle);

	void deletePlaces(int handle);
	int getNumPlacePtrs(int handle);
	int countDevAgents(int handle);

	/*
	 * Agent Mutators
	 */
	void* getAgentState(int handle);

	template<typename P, typename S>
	Place** instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	DeviceConfig(const DeviceConfig& other); // copy constructor
	DeviceConfig &operator=(const DeviceConfig &rhs); // assignment operator

private:
	int deviceNum;
//	cudaStream_t inputStream;
//	cudaStream_t outputStream;
//	cudaEvent_t deviceEvent;
//	PlaceArray devPlaces;
	std::map<int, AgentArray> devAgents;
	std::map<int, PlaceArray> devPlacesMap;

};
// end class

template<typename PlaceType>
__global__ void instantiatePlacesKernel(Place** places, void *state,
		int stateBytes, void *arg, int *dims, int nDims, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		// set pointer to corresponding state object
		char *tmpPtr = ((char*) state) + (idx * stateBytes);

		places[idx] = new PlaceType((PlaceState*) tmpPtr, arg);
		places[idx]->setIndex(idx);
		places[idx]->setSize(dims, nDims);
	}
}

template<typename P, typename S>
Place** DeviceConfig::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty) {

	if (devPlacesMap.count(handle) > 0) {
		return NULL;
	}

	setAsActiveDevice();

	// create places tracking
	PlaceArray p;
	p.qty = qty;

	// create state array on device
	void* tmp = NULL;
	CATCH(cudaMalloc((void** ) &tmp, qty * sizeof(S)));
	p.devState = tmp;

	// allocate device pointers
	Place** tmpPlaces = NULL;
	CATCH(cudaMalloc((void** ) &tmpPlaces, qty * sizeof(Place*)));
	p.devPtr = tmpPlaces;

	// handle arg
	void *d_arg = NULL;
	if (NULL != argument) {
		CATCH(cudaMalloc((void** ) &d_arg, argSize));
		CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
	}

	// load places dimensions
	int *d_dims;
	int dimBytes = sizeof(int) * dimensions;
	CATCH(cudaMalloc((void** ) &d_dims, dimBytes));
	CATCH(cudaMemcpy(d_dims, size, dimBytes, H2D));

	// launch instantiation kernel
	int blockDim = (qty - 1) / BLOCK_SIZE + 1;
	int threadDim = (qty - 1) / blockDim + 1;
	instantiatePlacesKernel<P> <<<blockDim, threadDim>>>(p.devPtr, p.devState,
			sizeof(S), d_arg, d_dims, dimensions, qty);
	CHECK();

	// clean up memory
	if (NULL != argument) {
		CATCH(cudaFree(d_arg));
	}
	CATCH(cudaFree(d_dims));

	devPlacesMap[handle] = p;
	return p.devPtr;
}

} // end namespace
