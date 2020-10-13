#pragma once

#include <map>
#include <cassert>
#include <unordered_set>

#include "iostream"
#include "cudaUtil.h"
#include "Logger.h"
#include "GlobalConsts.h"
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
	DeviceConfig(std::vector<int> devices);

	virtual ~DeviceConfig();
	void freeDevice();

	void load(void*& destination, const void* source, size_t bytes);
	void unload(void* destination, void* source, size_t bytes);

	Place** getDevPlaces(int handle);
	void* getPlaceState(int handle);
	int countDevPlaces(int handle);

	Agent** getDevAgents(int handle);
	void* getAgentsState(int handle);

	int getNumAgents(int handle);
	int getNumAgentObjects(int handle);
	int getMaxAgents(int handle);

	int getDeviceNum();
	// Returns block and thread dimentions for the Agent collection 
	//   based on the number of elements belonging to the collection
	
	dim3* getBlockThreadDims(int agentHandle);
	int* DeviceConfig::getSize();
	int DeviceConfig::getDims();
	void DeviceConfig::setSize(int *size);
	void DeviceConfig::setDims(int dims);
	std::vector<int> DeviceConfig::getDevices();

	std::vector<int> getDevices();
	// void setDevices(std::vector<int> devices);

	template<typename P, typename S>
	Place** instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	Agent** instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs);

private:
	// int deviceNum;
	int *size;
	int dimensions;
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

inline void getRandomPlaceIdxs(int idxs[], int nPlaces, int nAgents, int stride) {
	int curAllocated = 0;

	if (nAgents > nPlaces) {
		for (int i=0; i<nPlaces; i++) {
			for (int j=0; j<nAgents/nPlaces; j++) {
				idxs[curAllocated] = i;
				curAllocated++;
			}
		}
	}

	// Allocate the remaining agents randomly
	std::unordered_set<int> occupied_places;

	while (curAllocated < nAgents) {
		unsigned int randPlace = rand() % nPlaces + stride; //random number from 0 to nPlaces
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

	// set places dimensions
	this->size = size;
	this->dimensions = dimensions;

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

	// launch instantiation kernel
	int blockDim = (qty - 1) / BLOCK_SIZE + 1;
	int threadDim = (qty - 1) / blockDim + 1;
	int stride = qty / activeDevices.size();
	for (int i = 0; i < activeDevices.size(); ++i) {
		cudaSetDevice(activeDevices.at(i));

		// handle arg - puts argument on each device
		void *d_arg = NULL;
		if (NULL != argument) {
			CATCH(cudaMalloc((void** ) &d_arg, argSize));
			CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
		}

		// load places dimensions - puts places dimensions on each device
		int *d_dims;
		int dimBytes = sizeof(int) * dimensions;
		CATCH(cudaMalloc((void** ) &d_dims, dimBytes));
		CATCH(cudaMemcpy(d_dims, size, dimBytes, H2D));

		Logger::debug("Launching instantiation kernel on device: %d", activeDevices.at(i));
		instantiatePlacesKernel<P, S> <<<blockDim, threadDim>>>(p.devPtr + i * stride,
				d_state + i * stride, d_arg, d_dims, dimensions, stride);
		CHECK();
		CATCH(cudaFree(d_arg));
		CATCH(cudaFree(d_dims));
	}

	Logger::debug("Finished instantiation kernel");

	CATCH(cudaMemGetInfo(&freeMem, &allMem));

	devPlacesMap[handle] = p;
	Logger::debug("Finished DeviceConfig::instantiatePlaces");
	return p.devPtr;
}

template<typename AgentType, typename AgentStateType>
__global__ void instantiateAgentsKernel(Agent** agents, AgentStateType *state,
		void *arg, int nAgents, int maxAgents) {
	unsigned idx = getGlobalIdx_1D_1D(); // does this work for MGPU?

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

	PlaceArray places = devPlacesMap[placesHandle];
	int placesStride = places.qty / activeDevices.size();

	// If no agent mapping provided, split agents evenly amongst devices/place chunks and randomly generate placeIdxs
	int *agtDevArr[activeDevices.size()];
	int agtDevCount[activeDevices.size()];
	int strideSum = 0;
	if (placeIdxs == NULL) {
		for (int i = 0; i < activeDevices.size(); ++i) {
			agtDevCount[i] = nAgents / activeDevices.size();
			int tempPlaceIdxs[agtDevCount[i]];
			getRandomPlaceIdxs(&tempPlaceIdxs, placesStride, agtDevCount[i], strideSum);
			agtDevArr[i] = tempPlaceIdxs;
			strideSum += agtDevCount[i];
		}
	}

	// If provided a map of agents split by placeIdx as mapped to devices
	else {
		std::sort(placeIdxs, sizeof(placeIdxs) / sizeof(placeIdxs[0]));
		int count = 0;
		int *ptr_end = placeIdxs;
		int *ptr_begin = placeIdxs;
		for (int i = 0; i < activeDevices.size(); ++i) {
			while (*ptr_end < (i + 1) * placesStride) {
				ptr_end += sizeof(int);
				count++;
			}

			agtDevArr[i] = ptr_begin;
			agtDevCount[i] = count;
			ptr_begin += count * sizeof(int);
			count = 0;
		}
	}
	
	// launch map kernel using 1 thread per agent 
	if (nAgents / places.qty + 1 > MAX_AGENTS) {
		throw MassException("Number of agents per places exceeds the maximum setting 
			of the library. Please change the library setting MAX_AGENTS and 
			re-compile the library.");
	}

	// launch instantiation kernel
	int blockDim = (a.nObjects - 1) / BLOCK_SIZE + 1;
	int threadDim = (a.nObjects - 1) / blockDim + 1;

	int strideCount = 0;

	for (int i = 0; i < activeDevices.size(); ++i) {
		Logger::debug("Launching agent instantiation kernel on device: %d", activeDevices.at(i));
		cudaSetDevice(activeDevices.at(i));

		// handle arg on each device
		void *d_arg = NULL;
		if (NULL != argument) {
			CATCH(cudaMalloc((void** ) &d_arg, argSize));
			CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
		}

		instantiateAgentsKernel<AgentType, AgentStateType> 
				<<<blockDim, threadDim>>>(a.devPtr + strideCount, d_state + strideCount,
				temp_arg, agtDevCount[i], a.nObjects);
		CHECK();
		if (NULL != argument) {
			CATCH(cudaFree(d_arg));
		}

		strideCount += agtDevCount[i];
	}

	Logger::debug("Finished agent instantiation kernel");

	// Loop over devices and map agents to places on each device
	strideCount = 0;
	for (int i = 0; i < activeDevices.size(); ++i) {
		int* placeIdxs_d;
		CATCH(cudaMalloc(&placeIdxs_d, agtDevCount[i] * sizeof(int)));
		CATCH(cudaMemcpy(placeIdxs_d, agtDevArr[i], agtDevCount[i] * sizeof(int), H2D));

		Logger::debug("Launching agent mapping kernel on device: %d", activeDevices.at(i));
		mapAgentsKernel<AgentType, AgentStateType> <<<blockDim, threadDim>>>(places.devPtr + strideCount, 
				agtDevCount[i], a.devPtr, d_state, agtDevCount[i], placeIdxs_d);
		CHECK();

		CATCH(cudaFree(placeIdxs_d));

		strideCount += agtDevCount[i];
	}

	CATCH(cudaMemGetInfo(&freeMem, &allMem));

	devAgentsMap[handle] = a;
	Logger::debug("Finished DeviceConfig::instantiateAgents");
	return a.devPtr;
}

} // end namespace
