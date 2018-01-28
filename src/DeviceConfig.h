
#pragma once

#include <map>
#include <cassert>

#include "iostream"
#include "cudaUtil.h"
#include "Logger.h"
//#include "PlaceState.h"
#include "DataModel.h"
#include "GlobalConsts.h"

namespace mass {

// forward declarations
class Place;
class Agent;

// PlaceArray stores place pointers and state pointers on GPU
struct PlaceArray {
	Place** devPtr;
	void *devState;
	int qty;  //size
};

// PlaceArray stores place pointers and state pointers on GPU
struct AgentArray {
	Agent** devPtr;
	void *devState;
	int qty;  //size
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

	void load(void*& destination, const void* source, size_t bytes);
	void unload(void* destination, void* source, size_t bytes);

	/*
	 * Place Mutators
	 */

	Place** getDevPlaces(int handle);
	void* getPlaceState(int handle);
	int countDevPlaces(int handle);

	void deletePlaces(int handle);
	int getNumPlacePtrs(int handle);

	GlobalConsts getGlobalConstants();
	void updateConstants(GlobalConsts consts);

	template<typename P, typename S>
	Place** instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	Agent** instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents);

private:
	int deviceNum;
	std::map<int, PlaceArray> devPlacesMap;
	std::map<int, AgentArray> devAgentsMap;
	GlobalConsts glob;
	GlobalConsts *d_glob;
	size_t freeMem;
	size_t allMem;
};
// end class

template<typename PlaceType, typename StateType>
__global__ void instantiatePlacesKernel(Place** places, StateType *state,
		void *arg, int *dims, int nDims, int qty) {
	unsigned idx = getGlobalIdx_1D_1D();

	if (idx < qty) {
		// set pointer to corresponding state object
		places[idx] = new PlaceType(&(state[idx]), arg);
		places[idx]->setIndex(idx);
		places[idx]->setSize(dims, nDims);
	}
}

template<typename P, typename S>
Place** DeviceConfig::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty) {

	Logger::debug("Entering DeviceConfig::instantiatePlaces\n");

	if (devPlacesMap.count(handle) > 0) {
		return NULL;
	}

	// add global constants to the GPU
	memcpy(glob.globalDims, size, sizeof(int) * dimensions);
	updateConstants(glob);

	// create places tracking
	PlaceArray p;
	p.qty = qty; //size

	// create state array on device
	S* d_state = NULL;
	int Sbytes = sizeof(S);
	CATCH(cudaMalloc((void** ) &d_state, qty * Sbytes));
	p.devState = d_state;

	// allocate device pointers
	Place** tmpPlaces = NULL;
	int ptrbytes = sizeof(Place*);
	CATCH(cudaMalloc((void** ) &tmpPlaces, qty * ptrbytes));
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
	Logger::debug("Launching instantiation kernel");
	instantiatePlacesKernel<P, S> <<<blockDim, threadDim>>>(p.devPtr, d_state,
			d_arg, d_dims, dimensions, qty);
	CHECK();
	Logger::debug("Finished instantiation kernel");
	
	// clean up memory
	if (NULL != argument) {
		CATCH(cudaFree(d_arg));
	}
	CATCH(cudaFree(d_dims));
	CATCH(cudaMemGetInfo(&freeMem, &allMem));

	devPlacesMap[handle] = p;
	Logger::debug("Finished DeviceConfig::instantiatePlaces");
	return p.devPtr;
}

template<typename AgentType, typename AgentStateType>
__global__ void instantiateAgentsKernel(Agent** agents, AgentStateType *state,
		void *arg, int qty) {
	unsigned idx = getGlobalIdx_1D_1D();

	if (idx < qty) {
		// set pointer to corresponding state object
		agents[idx] = new AgentType(&(state[idx]), arg);
		agents[idx]->setIndex(idx);
		agents[idx]->setSize(qty);
		//TODO: link to Place*
	}
}

template<typename AgentType, typename AgentStateType>
Agent** DeviceConfig::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents) {
	Logger::debug("Entering DeviceConfig::instantiateAgents\n");

	if (devAgentsMap.count(handle) > 0) {
		return NULL;
	}

	// create places tracking
	AgentArray a;
	a.qty = nAgents; //size

	// create state array on device
	AgentStateType* d_state = NULL;
	int Sbytes = sizeof(AgentStateType);
	CATCH(cudaMalloc((void** ) &d_state, nAgents * Sbytes));
	a.devState = d_state;

	// allocate device pointers
	Agent** tmpAgents = NULL;
	int ptrbytes = sizeof(Agent*);
	CATCH(cudaMalloc((void** ) &tmpAgents, nAgents * ptrbytes));
	a.devPtr = tmpAgents;

	// handle arg
	void *d_arg = NULL;
	if (NULL != argument) {
		CATCH(cudaMalloc((void** ) &d_arg, argSize));
		CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
	}

	// launch instantiation kernel
	int blockDim = (nAgents - 1) / BLOCK_SIZE + 1;
	int threadDim = (nAgents - 1) / blockDim + 1;
	Logger::debug("Launching agent instantiation kernel");
	instantiateAgentsKernel<AgentType, AgentStateType> <<<blockDim, threadDim>>>(a.devPtr, d_state,
			d_arg, nAgents);
	CHECK();
	Logger::debug("Finished agent instantiation kernel");
	
	// clean up memory
	if (NULL != argument) {
		CATCH(cudaFree(d_arg));
	}
	CATCH(cudaMemGetInfo(&freeMem, &allMem));

	devAgentsMap[handle] = a;
	Logger::debug("Finished DeviceConfig::instantiateAgents");
	return a.devPtr;
}

} // end namespace
