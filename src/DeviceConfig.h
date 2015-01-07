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

namespace mass {

// forward declarations
class Agent;
class Place;

struct PlaceArray {
	Place** devPtr;
	int qty;
	PlaceArray() {
		devPtr = NULL;
		qty = 0;
	}
};

struct AgentArray {
	Agent** devPtr;
	int qty;
	AgentArray() {
		devPtr = NULL;
		qty = 0;
	}
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

	// void loadPlaces(PlacesPartition *part);
	// void loadAgents(AgentsPartition *part);

	bool isLoaded();
	void setLoaded(bool loaded);

	void setNumPlaces(int numPlaces);

	Place** getPlaces(int rank);
	int getNumPlacePtrs(int rank);

	DeviceConfig(const DeviceConfig& other); // copy constructor
	DeviceConfig &operator=(const DeviceConfig &rhs); // assignment operator

	template<class T>
	Place** instantiatePlaces(T instantiator, void* arg, int argSize,
			int handle, int qty);

	void deletePlaces(Place **places, int qty);

private:
	int deviceNum;
//	cudaStream_t inputStream;
//	cudaStream_t outputStream;
//	cudaEvent_t deviceEvent;
//	PlaceArray devPlaces;
	std::map<int, AgentArray> devAgents;
	std::map<int, PlaceArray> devPlacesMap;
	bool loaded;

};
// end class

template<class T>
__global__ void instantiatePlacesKernel(T* instantiator, Place** places,
		void *arg, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		places[idx] = instantiator->instantiate(arg);
	}
}

template<class T>
Place** DeviceConfig::instantiatePlaces(T instantiator, void* arg, int argSize,
		int handle, int qty) {
	if (devPlacesMap.count(handle) > 0) {
		return NULL;
	}
	setAsActiveDevice();

	// create places tracking
	PlaceArray p;
	p.qty = qty;
	devPlacesMap[handle] = p;

	Logger::warn("Attempting lethal cudaMalloc on %s %d", __FILE__, __LINE__);
	// allocate device pointers
	CATCH(cudaMalloc((void** ) &p.devPtr, qty * sizeof(Place*)));

	// copy instantiator to device
	T *d_inst; // device side instantiator
	CATCH(cudaMalloc((void** ) &d_inst, sizeof(T)));
	CATCH(cudaMemcpy(d_inst, &instantiator, sizeof(T), H2D));

	// handle arg
	void *d_arg = NULL;
	if (NULL != arg) {
		CATCH(cudaMalloc((void** ) &d_arg, argSize));
		CATCH(cudaMemcpy(d_arg, arg, argSize, H2D));
	}

	// launch instantiation kernel
	int blockDim = (qty - 1) / BLOCK_SIZE + 1;
	int threadDim = (qty - 1) / blockDim + 1;
	instantiatePlacesKernel<T> <<<blockDim, threadDim>>>(d_inst, p.devPtr, arg, qty);
	CHECK();

	// clean up memory
	CATCH(cudaFree(d_inst));
	if(NULL != d_arg){
		CATCH(cudaFree(d_arg));
	}

	return p.devPtr;
}

} // end namespace
