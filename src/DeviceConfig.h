#pragma once

#include <map>
#include <cassert>
#include <unordered_set>

#include "iostream"
#include "cudaUtil.h"
#include "Logger.h"
#include "DataModel.h"
// #include "GlobalConsts.h"
#include "MassException.h"

namespace mass {

// forward declaration
class Place;
class Agent;

// PlaceArray stores place pointers and state pointers on GPU
struct PlaceArray {
	Place** devPtr;
	void *devState;
	int qty;  //size
};

// AgentArray stores agent pointers and state pointers on GPU
struct AgentArray {
	Agent** devPtr;
	void *devState;
	int nAgents;  //number of live agents
	int nObjects; //number of all agent objects
	int nextIdx;  //next available idx for allocation
	dim3 dims[2]; //block and thread dimensions
};

/**
 *  This class represents a GPU.
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

	Place** getDevPlaces(int handle);
	void* getPlaceState(int handle);
	int countDevPlaces(int handle);

	int getNumPlacePtrs(int handle);

	Agent** getDevAgents(int handle);
	void* getAgentsState(int handle);

	int getNumAgents(int handle);
	int getNumAgentObjects(int handle);
	int getMaxAgents(int handle);

	int getDeviceNum();
	// Returns block and thread dimentions for the Agent collection 
	//   based on the number of elements beloging to the collection
	
	dim3* getDims(int agentHandle);

	template<typename P, typename S>
	Place** instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	Agent** instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs);

private:
	int deviceNum;
	std::vector<int> activeDevices;
	std::map<int, PlaceArray> devPlacesMap;
	std::map<int, AgentArray> devAgentsMap;

	// TODO: Add Agents and Places partitioning?

	size_t freeMem;
	size_t allMem;

	void deletePlaces(int handle);
	void deleteAgents(int handle);
};
// end class

inline void getRandomPlaceIdxs(int idxs[], int nPlaces, int nAgents) {
	int curAllocated = 0;

	if (nAgents > nPlaces) {
		for (int i=0; i<nPlaces; i++) {
			for (int j=0; j<nAgents/nPlaces; j++) {
				idxs[curAllocated] = i;
				curAllocated ++;
			}
		}
	}

	// Allocate the remaining agents randomly:
	std::unordered_set<int> occupied_places;

	while (curAllocated < nAgents) {
		unsigned int randPlace = rand() % nPlaces; //random number from 0 to nPlaces
		if (occupied_places.count(randPlace)==0) {
			occupied_places.insert(randPlace);
			idxs[curAllocated] = randPlace;
			curAllocated++;
		}
	}
}

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

	// create places tracking
	PlaceArray p;
	p.qty = qty; //size

	// create state array on device
	S* d_state = NULL;
	int Sbytes = sizeof(S);
	CATCH(cudaMallocManaged((void** ) &d_state, qty * Sbytes));
	p.devState = d_state;

	// allocate device pointers
	Place** tmpPlaces = NULL;
	int ptrbytes = sizeof(Place*);
	CATCH(cudaMallocManaged((void** ) &tmpPlaces, qty * ptrbytes));
	p.devPtr = tmpPlaces;

	// handle arg
	void *d_arg = NULL;
	if (NULL != argument) {
		CATCH(cudaMallocManaged((void** ) &d_arg, argument, argSize));
		//CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
	}

	// load places dimensions
	int *d_dims;
	int dimBytes = sizeof(int) * dimensions;
	CATCH(cudaMallocManaged((void** ) &d_dims, dimBytes));
	//CATCH(cudaMemcpy(d_dims, size, dimBytes, H2D));

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
		void *arg, int nAgents, int maxAgents) {
	unsigned idx = getGlobalIdx_1D_1D();

	if (idx < nAgents) {
		// set pointer to corresponding state object
		agents[idx] = new AgentType(&(state[idx]), arg);
		agents[idx]->setIndex(idx);
		agents[idx]->setAlive();
	} else if (idx < maxAgents) {
		//create placeholder objects for future agent spawning
		agents[idx] = new AgentType(&(state[idx]), arg);
	}
}

template<typename AgentType, typename AgentStateType>
__global__ void mapAgentsKernel(Place **places, int placeQty, Agent **agents,
		AgentStateType *state, int nAgents, int* placeIdxs) {
	int idx = getGlobalIdx_1D_1D();  //agent index

	if (idx < nAgents) {
		int placeIdx = placeIdxs[idx];
		Place* myPlace = places[placeIdx];
		agents[idx] -> setPlace(myPlace);
		myPlace -> addAgent(agents[idx]);
	}
}

template<typename AgentType, typename AgentStateType>
Agent** DeviceConfig::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs) {
	Logger::debug("Entering DeviceConfig::instantiateAgents\n");

	if (devAgentsMap.count(handle) > 0) {
		return NULL;
	}

	// create places tracking
	AgentArray a;
	a.nAgents = nAgents; //size
	if (maxAgents == 0) {
		a.nObjects = nAgents*2; //allocate more space to allow for agent spawning
	} else {
		a.nObjects = maxAgents;
	}
	
	a.nextIdx = nAgents;

	// create state array on device
	AgentStateType* d_state = NULL;
	int Sbytes = sizeof(AgentStateType);
	CATCH(cudaMallocManaged((void** ) &d_state, a.nObjects * Sbytes));
	a.devState = d_state;

	// allocate device pointers
	Agent** tmpAgents = NULL;
	int ptrbytes = sizeof(Agent*);
	CATCH(cudaMallocManaged((void** ) &tmpAgents, a.nObjects * ptrbytes));
	a.devPtr = tmpAgents;

	// handle arg
	void *d_arg = NULL;
	if (NULL != argument) {
		CATCH(cudaMallocManaged((void** ) &d_arg, argSize));
		//CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
	}

	// launch instantiation kernel
	int blockDim = (a.nObjects - 1) / BLOCK_SIZE + 1;
	int threadDim = (a.nObjects - 1) / blockDim + 1;
	Logger::debug("Launching agent instantiation kernel");
	instantiateAgentsKernel<AgentType, AgentStateType> <<<blockDim, threadDim>>>(a.devPtr, d_state,
			d_arg, a.nAgents, a.nObjects);
	CHECK();
	Logger::debug("Finished agent instantiation kernel");

	PlaceArray places = devPlacesMap[placesHandle];

	// if user did not provide a map of agents - create the default distribution of agents over places
	if (placeIdxs == NULL) {
		placeIdxs = new int[nAgents];
		getRandomPlaceIdxs(placeIdxs, places.qty, nAgents);
	}
	
	int* placeIdxs_d;
	CATCH(cudaMallocManaged(&placeIdxs_d, nAgents*sizeof(int)));
	//CATCH(cudaMemcpy(placeIdxs_d, placeIdxs, nAgents*sizeof(int), H2D));
	delete []placeIdxs;

	// launch map kernel using 1 thread per agent 
	if (nAgents / places.qty + 1 > MAX_AGENTS) {
		throw MassException("Number of agents per places exceeds the maximum setting of the library. Please change the library setting MAX_AGENTS and re-compile the library.");
	}
	Logger::debug("Launching agent mapping kernel");
	mapAgentsKernel<AgentType, AgentStateType> <<<blockDim, threadDim>>>(places.devPtr, places.qty,
			a.devPtr, d_state, nAgents, placeIdxs_d);
	CHECK();
	
	// clean up memory
	CATCH(cudaFree(placeIdxs_d));
	if (NULL != argument) {
		CATCH(cudaFree(d_arg));
	}
	CATCH(cudaMemGetInfo(&freeMem, &allMem));

	devAgentsMap[handle] = a;
	Logger::debug("Finished DeviceConfig::instantiateAgents");
	return a.devPtr;
}

} // end namespace
