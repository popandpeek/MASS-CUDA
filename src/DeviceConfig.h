#pragma once

#include <map>
#include <cassert>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "iostream"
#include "cudaUtil.h"
#include "Logger.h"
#include "GlobalConsts.h"
#include "MassException.h"
#include "settings.h"


namespace mass {

// forward declaration
class Place;
class Agent;
class AgentState;
class PlaceState;

// PlaceArray stores place pointers and state pointers on GPU
struct PlaceArray {
	std::vector<Place**> devPtrs;
	std::vector<void*> devStates;
	std::vector<int*> devDims; //dimension size of each device Places chunk 
	int qty;  //size
	int placesStride;
	int* ghostSpaceMultiple; //multipled by dimSize[0] and MAX_AGENT_TRAVEL to find num of ghost places on each device
	dim3 pDims[2];
};

// AgentArray stores agent pointers and state pointers on GPU
struct AgentArray {
	std::vector<Agent**> devPtrs;
	std::vector<void*> devStates;
	std::vector<Agent**> collectedAgents;

	int nAgents;  //number of live agents
	int* maxAgents; //number of all agent objects
	int* nAgentsDev; // tracking array for agents on each device
	dim3 aDims[2]; //block and thread dimensions
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

	std::vector<Place**> getDevPlaces(int handle);
	std::vector<void*> getPlaceStates(int handle);
	int getPlaceCount(int handle);
	void setPlacesThreadBlockDims(int handle);
	dim3* getPlacesThreadBlockDims(int handle);
	int getPlacesStride(int handle);
	int* getGhostPlaceMultiples(int handle);

	std::vector<Agent**> getDevAgents(int handle);
	std::vector<void*> getAgentsState(int handle);

	int getNumAgents(int handle);
	int getMaxAgents(int handle, int device);
	int* getMaxAgents(int handle);
	void setAgentsThreadBlockDims(int handle);
	dim3* getAgentsThreadBlockDims(int handle);
	int* getnAgentsDev(int handle);
	std::vector<Agent**> getBagOAgentsDevPtrs(int agentHandle);

	// returns machine number for device in held by default numbering in vector
	int getDeviceNum(int device);

	// returns the number of GPUS in the local system
	int getNumDevices();

	int* getDimSize();
	int getDimensions();
	void setDimSize(int *size);
	void setDimensions(int dims);
	std::vector<int> getDevices();

	template<typename P, typename S>
	std::vector<Place**> instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	std::vector<Agent**> instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs);

private:
	int *dimSize;
	int dimensions;
	std::vector<int> activeDevices;
	std::map<int, PlaceArray> devPlacesMap;
	std::map<int, AgentArray> devAgentsMap;

	size_t freeMem;
	size_t allMem;

	void deletePlaces(int handle);
	void deleteAgents(int handle);
};
// end class

inline void getRandomPlaceIdxs(int* idxs, int nPlaces, int nAgents) {
	int curAllocated = 0;
	// Logger::debug("DeviceConfig::getRandomPlaceIdxs: nPlaces == %d, nAgents == %d, stride == %d",
	// 		nPlaces, nAgents, stride);
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
		unsigned int randPlace = rand() % nPlaces; //random number from 0 to nPlaces
		if (occupied_places.count(randPlace)==0) {
			occupied_places.insert(randPlace);
			idxs[curAllocated] = randPlace;
			// Logger::debug("DeviceConfig::getRandomPlaceIdxs: idxs[%d] == %d", curAllocated, randPlace);
			++curAllocated;
		}
	}
}

template<typename PlaceType, typename StateType>
__global__ void instantiatePlacesKernel(Place** places, StateType *state,
		void *arg, int *dims, int *devDims, int nDims, int qty, int ghostPlaceQty,
		int device, int flip) {

	unsigned idx = getGlobalIdx_1D_1D();

	if (idx < qty) {
		// set pointer to corresponding state object
		int pIdx = idx + (flip * ghostPlaceQty);
		places[pIdx] = new PlaceType(&(state[idx]), arg);
		places[pIdx]->setDevIndex(idx);
		places[pIdx]->setIndex(idx + (device * qty));
		places[pIdx]->setSize(dims, devDims, nDims);
	}
}

template<typename PlaceType, typename StateType>
__global__ void instantiateGhostPlacesKernel(Place** places, StateType* state, 
		int qty, int placeStride, int stateStride) {

	unsigned idx = getGlobalIdx_1D_1D();
	if (idx < qty) {
		places[idx + placeStride] = new PlaceType();
		places[idx + placeStride]->setState(&(state[idx + stateStride]));
	}
}

template<typename P, typename S>
std::vector<Place**> DeviceConfig::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty) {

	Logger::debug("Entering DeviceConfig::instantiatePlaces\n");
	Logger::debug("Places size == size[0] = %d : size[1] = %d", size[0], size[1]);
	if (devPlacesMap.count(handle) > 0) {
		Logger::debug("DeviceConfig::instantiatePlaces: Places already there.");
		return {};
	}

	int numDevices = activeDevices.size();
	Logger::debug("Number of active devices == %d", activeDevices.size());
	
	// create places tracking
	PlaceArray p;
	p.qty = qty; //size
	p.placesStride = qty / activeDevices.size();

	// set place ghost spacing for each device
	p.ghostSpaceMultiple = new int[activeDevices.size()];
	for (int i = 0; i < activeDevices.size(); ++i) {
		if (i == 0 || i == activeDevices.size() - 1) {
			p.ghostSpaceMultiple[i] = 1 * MAX_AGENT_TRAVEL;
		}	

		else {
			p.ghostSpaceMultiple[i] = 2 * MAX_AGENT_TRAVEL;
		}
	}

	// Calculates the size of one ghost row
	int ghostRowSize = 1;
	for (int i = 1; i < dimensions; ++i) {
		ghostRowSize *= size[i];
	}

	// calculates the dimensional size of each devices places chunk
	for (int i = 0; i < activeDevices.size(); i++) {
		int* chunkSize = new int[dimensions];
		for (int j = 0; j < dimensions - 1; j++) {
			chunkSize[j] = size[j];
		}
		if (i != activeDevices.size() - 1) {
			chunkSize[dimensions - 1] = size[dimensions - 1] / activeDevices.size() + p.ghostSpaceMultiple[i];
		}
		else {
			chunkSize[dimensions - 1] = size[dimensions - 1] / activeDevices.size();
		}
		p.devDims.push_back(chunkSize);
		Logger::debug("chunkSize for device: %d == %d X %d", i, chunkSize[0], chunkSize[1]);
	}
	
	Logger::debug("ghostRowSize = %d", ghostRowSize);

	// set places dimensions
	this->setDimensions(dimensions);
	this->setDimSize(size);

	// create state vector of arrays to represent data on each device
	int Sbytes = sizeof(S);
	for (int i = 0; i < activeDevices.size(); ++i) {
		S* d_state = NULL;
		CATCH(cudaMalloc(&d_state, p.placesStride * Sbytes));
		Logger::debug("DeviceConfig::instantiatePlaces: size of d_state = %d; size of sbytes = %d; placesStride = %d", sizeof(*d_state), Sbytes, p.placesStride);
		p.devStates.push_back((d_state));
	}

	Logger::debug("DeviceConfig::InstantiatePlaces: Places State loaded into vector.");

	// create place vector for device pointers on each device - includes ghost places
	int ptrbytes = sizeof(Place*);
	for (int i = 0; i < activeDevices.size(); ++i) {
		Place** tmpPlaces;
		CATCH(cudaMalloc(&tmpPlaces, (p.placesStride + (ghostRowSize * p.ghostSpaceMultiple[i])) * ptrbytes));
		p.devPtrs.push_back(tmpPlaces);
	}

	int blockDim = (p.placesStride + 2 * p.ghostSpaceMultiple[0] * ghostRowSize) / BLOCK_SIZE + 1;
	int threadDim = (p.placesStride + 2 * p.ghostSpaceMultiple[0] * ghostRowSize) / blockDim + 1;
	Logger::debug("Kernel dims = gridDim %d, and blockDim = %d, ", blockDim, threadDim);
	
	// int to ensure we don't put ghost places[idx] < 0 when assigning indices in kernel function
	int flip = 0;
	
	for (int i = 0; i < activeDevices.size(); ++i) {
		Logger::debug("Launching instantiation kernel on device: %d with params: placesStride = %d, ghostRowSize = %d, ghostMult = %d", activeDevices.at(i), p.placesStride, ghostRowSize, p.ghostSpaceMultiple[i]);
		CATCH(cudaSetDevice(activeDevices.at(i)));
		// handle arg 
		void *d_arg = NULL;
		if (NULL != argument) {
			CATCH(cudaMalloc((void** )&d_arg, argSize));
			CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
		}

		// load places dimensions 
		int *d_dims = NULL;
		int *d_devDims = NULL;
		int dimBytes = sizeof(int) * dimensions;
		CATCH(cudaMalloc((void** ) &d_dims, dimBytes));
		CATCH(cudaMalloc((void** ) &d_devDims, dimBytes));
		CATCH(cudaMemcpy(d_dims, this->getDimSize(), dimBytes, H2D));
		CATCH(cudaMemcpy(d_devDims, p.devDims.at(i), dimBytes, H2D));
		Logger::debug("DeviceConfig::instantiatePlace: placesStride = %d, ghostPlaceMult = %d, ghostRowSize = %d, device = %d, flip = %d", p.placesStride, 
				p.ghostSpaceMultiple[i], ghostRowSize, i, flip);
		instantiatePlacesKernel<P, S> <<<blockDim, threadDim>>>(p.devPtrs[i], 
				(S*)(p.devStates[i]), d_arg, d_dims, d_devDims, dimensions, 
				p.placesStride, ghostRowSize * MAX_AGENT_TRAVEL, i, flip);
		CHECK();
		if (NULL != argument) {
			CATCH(cudaFree(d_arg));
		}

		if (flip == 0) {
			flip = 1;
		}

		CATCH(cudaFree(d_dims));
	}

	cudaDeviceSynchronize();
	for (int i = 0; i < activeDevices.size(); ++i) {
		cudaSetDevice(activeDevices.at(i));
		Logger::debug("Launching ghost Places instantiation kernel on device: %d ", activeDevices.at(i));
		if (i % 2 == 0) {
			// do right ghost stripe
			instantiateGhostPlacesKernel<P, S><<<blockDim, threadDim>>>(p.devPtrs[i], (S*)(p.devStates[i + 1]), 
					ghostRowSize * MAX_AGENT_TRAVEL, p.placesStride + (ghostRowSize * p.ghostSpaceMultiple[i]) - (ghostRowSize * MAX_AGENT_TRAVEL), 0);
			if (i != 0) {
				// do left ghost stripe
				instantiateGhostPlacesKernel<P, S><<<blockDim, threadDim>>>(p.devPtrs[i], (S*)(p.devStates[i - 1]), 
						ghostRowSize * MAX_AGENT_TRAVEL, 0, p.placesStride - (ghostRowSize * MAX_AGENT_TRAVEL));
			}
		}

		else {
			// do left ghost stripe
			instantiateGhostPlacesKernel<P, S><<<blockDim, threadDim>>>(p.devPtrs[i], (S*)(p.devStates[i - 1]), 
					ghostRowSize * MAX_AGENT_TRAVEL, 0, p.placesStride - (ghostRowSize * MAX_AGENT_TRAVEL));
			if (i != activeDevices.size() - 1) {
				// do right ghost stripe
				instantiateGhostPlacesKernel<P, S><<<blockDim, threadDim>>>(p.devPtrs[i], (S*)(p.devStates[i + 1]), 
						ghostRowSize * MAX_AGENT_TRAVEL, p.placesStride + (ghostRowSize * MAX_AGENT_TRAVEL), 0);
			}
		}

	}

	Logger::debug("Finished instantiation kernel");

	CATCH(cudaMemGetInfo(&freeMem, &allMem));
	devPlacesMap[handle] = p;
	setPlacesThreadBlockDims(handle);
	return p.devPtrs;
}

template<typename AgentType, typename AgentStateType>
__global__ void instantiateAgentsKernel(Agent** agents, AgentStateType *state,
		void *arg, int nAgents, int maxAgents, int agentsPerDevSum) {
	unsigned idx = getGlobalIdx_1D_1D(); // does this work for MGPU?

	if ((idx < nAgents)) {
		// set pointer to corresponding state object
		agents[idx] = new AgentType(&(state[idx]), arg);
		agents[idx]->setIndex(idx); // + agentsPerDevSum);
		agents[idx]->setAlive();
	} else if (idx < maxAgents) {
		//create placeholder objects for future agent spawning
		agents[idx] = new AgentType(&(state[idx]), arg);
	}
}

template<typename AgentType, typename AgentStateType>
__global__ void mapAgentsKernel(Place **places, Agent **agents,
		AgentStateType *state, int nAgents, int* placeIdxs, int ghostPlaceChunkSize) {
	unsigned idx = getGlobalIdx_1D_1D();  //agent index

	if (idx < nAgents) {
		int placeIdx = placeIdxs[idx];
		Place* myPlace = places[placeIdx + ghostPlaceChunkSize];
		agents[idx] -> setPlace(myPlace);
		myPlace -> addAgent(agents[idx]);
	}
}

template<typename AgentType, typename AgentStateType>
std::vector<Agent**> DeviceConfig::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs) {
	Logger::debug("Entering DeviceConfig::instantiateAgents\n");
	if (devAgentsMap.count(handle) > 0 /* && devAgentsMap[handle].nAgents != NULL*/) {
		Logger::debug("DeviceConfig::instantiatePlaces: Agents already there.");
		return {};
	}

	// create places tracking
	AgentArray a;
	Logger::debug("DeviceConfig::instantiateAgents: number of agents: %d", nAgents);
	a.nAgents = nAgents;
	int nObjects = 0;
	if (maxAgents == 0) {
		nObjects = a.nAgents*2; //allocate more space to allow for agent spawning
	} else {
		nObjects = maxAgents;
	}

	a.nAgentsDev = new int[activeDevices.size()]{};
	a.maxAgents = new int[activeDevices.size()]{};

	PlaceArray places = devPlacesMap[placesHandle];

	Logger::debug("DeviceConfig::instantiateAgents() - Successfully loads (%d) places from memory.", places.qty);

	// If no agent mapping provided, split agents evenly amongst devices/place chunks and randomly generate placeIdxs
	int **agtDevArr = new int*[activeDevices.size()];

	int strideSum = 0;
	if (placeIdxs == NULL) {
		Logger::debug("DeviceConfig::instantiateAgents() - Begin of placeIdxs = NULL init (evenly splits agents between devices and over their set of places).");
		for (int i = 0; i < activeDevices.size(); ++i) {
			a.nAgentsDev[i] = a.nAgents / activeDevices.size();
			Logger::debug("DeviceConfig::instantiateAgents() - %d number of agents over %d number of places on device %d .", a.nAgentsDev[i], places.placesStride, activeDevices.at(i));	
			int *tempPlaceIdxs = new int[a.nAgentsDev[i]];
			getRandomPlaceIdxs(tempPlaceIdxs, places.placesStride, a.nAgentsDev[i]);
			Logger::debug("DeviceConfig::instantiateAgents() - Completes call to getRandomPlaceIdxs().");
			agtDevArr[i] = tempPlaceIdxs;
			for (int j = 0; j < a.nAgentsDev[i]; ++j) {
				// Logger::debug("DeviceConfig::instantiateAgents() - agtDevArr[%d] == %d", j, agtDevArr[i][j]);
			}

			strideSum += places.placesStride;
			a.maxAgents[i] = nObjects / activeDevices.size();
		}
		Logger::debug("DeviceConfig::instantiateAgents() - Successfully completes placeIdxs = NULL init (evenly splits agents between devices and over their set of places).");
	}

	// If provided a map of agents split by placeIdx as mapped to devices
	else {
		std::sort(placeIdxs, placeIdxs + a.nAgents * sizeof(int));
		int count = 0;
		int *ptr_end = placeIdxs;
		int *ptr_begin = placeIdxs;
		for (int i = 0; i < activeDevices.size(); ++i) {
			while (*ptr_end < (i + 1) * places.placesStride) {
				ptr_end += sizeof(int);
				count++;
			}

			agtDevArr[i] = ptr_begin;
			a.nAgentsDev[i] = count;
			ptr_begin += count;
			count = 0;
		}
		
		int maxAgentsToShare = maxAgents - a.nAgents;
		int maxAgtAcc = 0;
		for (int i = 0; i < activeDevices.size() - 1; ++i) {
			int tempSum = a.nAgentsDev[i] + (maxAgentsToShare / activeDevices.size());
			a.maxAgents[i] = tempSum;
			maxAgtAcc += tempSum;
		}

		a.maxAgents[activeDevices.size() - 1] = a.nAgentsDev[activeDevices.size() - 1] + (maxAgentsToShare - maxAgtAcc);

		Logger::debug("DeviceConfig::instantiateAgents() - Successfully completes placeIdxs != NULL init (split amongst devices based on providedd init location).");
	}
	
	// create state array on device
	int Sbytes = sizeof(AgentStateType);
	for (int i = 0; i < activeDevices.size(); ++i) {
		AgentStateType* d_state = NULL;
		CATCH(cudaMalloc((void** ) &d_state, a.maxAgents[i] * Sbytes));
		a.devStates.push_back(d_state);
	}

	// allocate device pointers
	int ptrbytes = sizeof(Agent*);
	for (int i = 0; i < activeDevices.size(); ++i) {
		Agent** tmpAgents = NULL;
		CATCH(cudaMalloc((void** ) &tmpAgents, a.maxAgents[i] * ptrbytes));
		a.devPtrs.push_back(tmpAgents);
	}

	Logger::debug("DeviceConfig::instantiateAgents() finshed with agent memory initialization.");

	// launch map kernel using 1 thread per agent 
	if (a.nAgents / places.qty + 1 > MAX_AGENTS) {
		throw MassException("Number of agents per places exceeds the maximum setting of the library. Please change the library setting MAX_AGENTS and re-compile the library.");
	}

	// launch instantiation kernel
	devAgentsMap[handle] = a;
	setAgentsThreadBlockDims(handle);
	dim3* aDims = getAgentsThreadBlockDims(handle);
	Logger::debug("DeviceConfig::instantiateAgents: nAgents = %d", a.nAgents);
	Logger::debug("Kernel dims = gridDim %d and blockDim = %d", aDims[0].x, aDims[1].x);
	int agentsPerDevSum = 0;
	for (int i = 0; i < activeDevices.size(); ++i) {
		Logger::debug("Launching agent instantiation kernel on device: %d", activeDevices.at(i));
		CATCH(cudaSetDevice(activeDevices.at(i)));

		// handle arg on each device
		void *d_arg = NULL;
		if (NULL != argument) {
			CATCH(cudaMalloc(&d_arg, argSize));
			CATCH(cudaMemcpy(d_arg, argument, argSize, H2D));
		}
		instantiateAgentsKernel<AgentType, AgentStateType> <<<aDims[0], aDims[1]>>>(a.devPtrs.at(i), 
			(AgentStateType*)a.devStates.at(i), d_arg, a.nAgentsDev[i], a.maxAgents[i], agentsPerDevSum);
		CHECK();
		if (NULL != argument) {
			CATCH(cudaFree(d_arg));
		}
		agentsPerDevSum += a.nAgentsDev[i];
	}

	Logger::debug("Finished agent instantiation kernel");

	// Loop over devices and map agents to places on each device
	int ghostPlaceChunkSize = 0;
	for (int i = 0; i < activeDevices.size(); ++i) {
		CATCH(cudaSetDevice(activeDevices.at(i)));
		int* placeIdxs_d;
		CATCH(cudaMalloc((void** )&placeIdxs_d, a.nAgentsDev[i] * sizeof(int)));
		CATCH(cudaMemcpy(placeIdxs_d, agtDevArr[i], a.nAgentsDev[i] * sizeof(int), H2D));
		Logger::debug("Launching agent mapping kernel on device: %d", activeDevices.at(i));
		Logger::debug("Size of p.devPtrs = %d; a.devPtrs = %d; a.devStates = %d", places.devPtrs.size(), a.devPtrs.size(), a.devStates.size());
		Logger::debug("Size of a.nAgentsDev[%d] = %d", i, a.nAgentsDev[i]);
		Logger::debug("ghostChunkPlaceSize = %d", ghostPlaceChunkSize);
		mapAgentsKernel<AgentType, AgentStateType> <<<aDims[0], aDims[1]>>>(places.devPtrs.at(i), 
				a.devPtrs.at(i), (AgentStateType*)(a.devStates.at(i)), 
				a.nAgentsDev[i], placeIdxs_d, ghostPlaceChunkSize);
		cudaDeviceSynchronize();
		CHECK();
		CATCH(cudaFree(placeIdxs_d));
		if (ghostPlaceChunkSize == 0) {
			ghostPlaceChunkSize = dimSize[0] * MAX_AGENT_TRAVEL;
		}
	}

	CATCH(cudaMemGetInfo(&freeMem, &allMem));
	Logger::debug("Finished DeviceConfig::instantiateAgents.\n");
	return a.devPtrs;
}

} // end namespace
