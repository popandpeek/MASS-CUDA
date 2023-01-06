
#include "DeviceConfig.h"
#include "Place.h"
#include "PlaceState.h"
#include "cudaUtil.h"
#include "Logger.h"
#include "MassException.h"
#include "string.h"
#include "settings.h"

using namespace std;

namespace mass {

DeviceConfig::DeviceConfig() :
		activeDevices(-1) {

	// TODO: Change to have on each device and fix destructor to delete on all devices
	// randStates = NULL;
	// randStateSize = NULL;
	Logger::warn("DeviceConfig::NoParam constructor");
}

DeviceConfig::DeviceConfig(std::vector<int> devices) {
	activeDevices = devices;
	devPlacesMap = map<int, PlaceArray>{};
	devAgentsMap = map<int, AgentArray>{};
	freeMem = new size_t[activeDevices.size()];
	allMem = new size_t[activeDevices.size()];
	limit = new size_t[activeDevices.size()];
	// randStates = new curandState*[activeDevices.size()];
	// randStateSize = new int[activeDevices.size()];
    #pragma omp parallel 
    {
		int gpu_id = -1;
		CATCH(cudaGetDevice(&gpu_id));
		// randStates[gpu_id] = NULL;
		// randStateSize[gpu_id] = 0;
		CATCH(cudaDeviceGetLimit(&limit[gpu_id], cudaLimitMallocHeapSize));
		CATCH(cudaMemGetInfo(&freeMem[gpu_id], &allMem[gpu_id]));
		Logger::debug("DeviceConfig: Constructor: device = %d and mem limit = %llu", gpu_id, limit[gpu_id]);
		Logger::debug("DeviceConfig: Constructor: device = %d and allMem = %llu and freeMem = %llu", gpu_id, allMem[gpu_id], freeMem[gpu_id]);
		size_t total =  size_t(2048) * size_t(2048) * size_t(1536);
		CATCH(cudaDeviceSetLimit(cudaLimitMallocHeapSize, total));
		CATCH(cudaDeviceGetLimit(&limit[gpu_id], cudaLimitMallocHeapSize));
		Logger::debug("DeviceConfig: Constructor: device = %d and increased mem limit = %llu", gpu_id, limit[gpu_id]);
    }
}

DeviceConfig::~DeviceConfig() {
	Logger::debug("deviceConfig destructor ");
}

void DeviceConfig::freeDevice() {
	Logger::debug("DeviceConfig::freeDevice()");
	Logger::debug("deviceConfig delete Agent's");
	std::map<int, AgentArray>::iterator it_a = devAgentsMap.begin();
	int agentsMapSize = devAgentsMap.size();
	while (it_a != devAgentsMap.end()) {
		deleteAgents(it_a->first);
		devAgentsMap.erase(it_a++);
	}
	devAgentsMap.clear();

	// Delete places:
	Logger::debug("deviceConfig delete Place's");
	std::map<int, PlaceArray>::iterator it_p = devPlacesMap.begin();
	int placesSize = devPlacesMap.size();
	while (it_p != devPlacesMap.end()) {
		deletePlaces(it_p->first);
		Logger::debug("DeviceConfig::freeDevice(): Returns from deletePlaces().");
		devPlacesMap.erase(it_p++);
	}

	// delete[] randStateSize;
	// for (int i = 0; i < activeDevices.size(); ++i) {
	// 	CATCH(cudaSetDevice(i));
	// 	if (randStates[i] != NULL) {
	// 		cudaFree(randStates[i]);
	// 	}
	// }

	/// delete[] randStates;

	devPlacesMap.clear();
	for (std::size_t i = 0; i < activeDevices.size(); ++i) {
		CATCH(cudaSetDevice(activeDevices.at(i)));
		CATCH(cudaDeviceReset());
	}

	delete freeMem, allMem, limit;
	Logger::debug("Done with deviceConfig freeDevice().");
}

// // TODO: Refactor for UVA memory - NO CHANGES NEEDED?
// void DeviceConfig::load(void*& destination, const void* source, size_t bytes) {
//	CATCH(cudaMalloc((void** ) &destination, bytes));
// 	CATCH(cudaMemcpy(destination, source, bytes, H2D));
// 	CATCH(cudaMemGetInfo(&freeMem, &allMem));
// }

// // TODO: Refactor for UVA memory
// void DeviceConfig::unload(void* destination, void* source, size_t bytes) {
// 	CATCH(cudaMemcpy(destination, source, bytes, D2H));
// 	CATCH(cudaFree(source));
// 	CATCH(cudaMemGetInfo(&freeMem, &allMem));
// }

std::vector<Place**> DeviceConfig::getDevPlaces(int handle) {
	return devPlacesMap[handle].devPtrs;
}

std::vector<std::pair<Place**, void*>> DeviceConfig::getTopNeighborGhostPlaces(int handle) {
	return devPlacesMap[handle].topNeighborGhosts;
}

std::vector<std::pair<Place**, void*>> DeviceConfig::getTopGhostPlaces(int handle) {
	return devPlacesMap[handle].topGhosts;
}

std::vector<std::pair<Place**, void*>> DeviceConfig::getBottomGhostPlaces(int handle) {
	return devPlacesMap[handle].bottomGhosts;
}

std::vector<std::pair<Place**, void*>> DeviceConfig::getBottomNeighborGhostPlaces(int handle) {
	return devPlacesMap[handle].bottomNeighborGhosts;
}

std::vector<void*> DeviceConfig::getPlaceStates(int handle) {
	return devPlacesMap[handle].devStates;
}

int DeviceConfig::getPlaceCount(int handle) {
	if (devPlacesMap.count(handle) != 1) {
		throw MassException("Handle not found.");
	}
	return devPlacesMap[handle].qty;
}

void DeviceConfig::setPlacesThreadBlockDims(int handle) {
	Logger::debug("DeviceConfig::setPlacesThreadBlockDims(): numPlaces == %d, numDevices == %d", devPlacesMap[handle].qty, activeDevices.size());
	int numBlocks = ((devPlacesMap[handle].placesStride + (2 * dimSize[0] * devPlacesMap[handle].ghostSpaceMultiple[0])) - 1) / BLOCK_SIZE + 1;
	int nThr = ((devPlacesMap[handle].placesStride + (2 * dimSize[0] * devPlacesMap[handle].ghostSpaceMultiple[0])) - 1) / numBlocks + 1;
	dim3 bDim = dim3(numBlocks);
	dim3 tDim = dim3(nThr);

	devPlacesMap[handle].pDims[0] = bDim;
	devPlacesMap[handle].pDims[1] = tDim;
	Logger::debug("setPlacesThreadBlockDims(): numBlocks == %u, %u, %u; nThr == %u, %u, %u", bDim.x, bDim.y, bDim.z, tDim.x, tDim.y, tDim.z);
}

dim3* DeviceConfig::getPlacesThreadBlockDims(int handle) {
	dim3* ret = &(*devPlacesMap[handle].pDims);
	Logger::debug("pDims[0]: { %d, %d, %d }, pDims[1]: { %d, %d, %d }", ret[0].x, ret[0].y, ret[0].z, ret[1].x, ret[1].y, ret[1].z);
	return ret;
}

int DeviceConfig::getPlacesStride(int handle) {
	return devPlacesMap[handle].placesStride;
}

int* DeviceConfig::getGhostPlaceMultiples(int handle) {
	return devPlacesMap[handle].ghostSpaceMultiple;
}

std::vector<Agent**> DeviceConfig::getDevAgents(int handle) {
	return devAgentsMap[handle].devPtrs;
}

std::vector<void*> DeviceConfig::getAgentsState(int handle) {
	return devAgentsMap[handle].devStates; 
}

int DeviceConfig::getNumAgents(int handle) {
	return devAgentsMap[handle].nAgents;
}

// int DeviceConfig::getMaxAgents(int handle, int device) {
// 	return devAgentsMap[handle].maxAgents[device];
// }

int DeviceConfig::getMaxAgents(int handle) {
	return devAgentsMap[handle].maxAgents;
}

void DeviceConfig::setAgentsThreadBlockDims(int handle) {
	// TODO: Need to update aDims to hold multiple sets
	Logger::debug("DeviceConfig::setAgentsMapThreadBlockDims(): numAgents == %d, numDevices == %d", devAgentsMap[handle].nAgents, activeDevices.size());
	
	for (int i = 0; i < activeDevices.size(); ++i) {
		Logger::debug("DeviceConfig::setAgentsMapThreadBlockDims(): maxAgents[%d] = %d", i, devAgentsMap[handle].maxAgents);
		int numBlocks = ((devAgentsMap[handle].maxAgents - 1) / BLOCK_SIZE) + 1;
		int nThr = ((devAgentsMap[handle].maxAgents - 1) / numBlocks) + 1;
		dim3 bDim = dim3(numBlocks, 1, 1);
		dim3 tDim = dim3(nThr, 1, 1);
		std::pair<dim3, dim3> temp = std::make_pair(bDim, tDim);
		devAgentsMap[handle].aDims.push_back(temp);
		Logger::debug("DeviceConfig::setAgentsThreadBlockDims: bDim = %d, tDim = %d", (devAgentsMap[handle].aDims.at(i).first).x, (devAgentsMap[handle].aDims.at(i).second).x);
	}
}

int* DeviceConfig::getnAgentsDev(int handle) {
	return devAgentsMap[handle].nAgentsDev;
}

void DeviceConfig::setnAgentsDev(int handle, int device, int nAgents) {
	devAgentsMap[handle].nAgentsDev[device] = nAgents;
}

std::vector<std::pair<dim3, dim3>> DeviceConfig::getAgentsThreadBlockDims(int handle) {
	return devAgentsMap[handle].aDims;
}

int DeviceConfig::getDeviceNum(int device) {
	return activeDevices.at(device);
}

int DeviceConfig::getNumDevices() {
	return activeDevices.size();
}

int DeviceConfig::getPlaceStateSize(int handle) {
	return devPlacesMap[handle].stateSize;
}

int DeviceConfig::getAgentStateSize(int handle) {
	return devAgentsMap[handle].stateSize;
}
__global__ void destroyAgentsKernel(Agent **agents, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		delete agents[idx];
	}
}

void DeviceConfig::deleteAgents(int handle) {
	Logger::debug("DeviceConfig:: Entering deleteAgents.");
	AgentArray a = devAgentsMap[handle];
	for (int i = 0; i < a.devPtrs.size(); ++i) {
		Logger::debug("DeviceConfig::deleteAgents: device: %d, Number to delete: %d.", i, a.maxAgents / a.devPtrs.size());
		cudaSetDevice(activeDevices.at(i));
		destroyAgentsKernel<<<a.aDims.at(i).first, a.aDims.at(i).second>>>(a.devPtrs.at(i), a.maxAgents / a.devPtrs.size());
		CHECK();
		cudaDeviceSynchronize();
		CATCH(cudaFree(a.devPtrs.at(i)));
		CATCH(cudaFree(a.devStates.at(i)));
		Logger::debug("DeviceConfig::deleteAgents: CUDA memory freed on device: %d", i);
	}

	delete a.nAgentsDev;
	Logger::debug("DeviceConfig::deleteAgents: SUCCESS!");
}

__global__ void destroyPlacesKernel(Place **places, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		delete places[idx];
	}
}

// TODO: NULL all helper pointers before the kernel
void DeviceConfig::deletePlaces(int handle) {
	Logger::debug("DeviceConfig:: Entering deletePlaces.");
	PlaceArray p = devPlacesMap[handle];
	Logger::debug("DeviceConfig::deletePlaces: Size of places map == %d", devPlacesMap.size());
	for (int i = 0; i < p.devPtrs.size(); ++i) {
		Logger::debug("DeviceConfig::deletePlaces: device: %d, Number to delete: %d.", i, p.placesStride + (p.ghostSpaceMultiple[i] * dimSize[0]));
		cudaSetDevice(activeDevices.at(i));
		destroyPlacesKernel<<<p.pDims[0], p.pDims[1]>>>(p.devPtrs.at(i), 
				(p.placesStride + (p.ghostSpaceMultiple[i] * dimSize[0])));
		CHECK();
		cudaDeviceSynchronize();
		CATCH(cudaFree(p.devPtrs.at(i)));
		CATCH(cudaFree(p.devStates.at(i)));
		Logger::debug("DeviceConfig::deletePlaces: CUDA memory freed on device: %d", i);
	}


	delete[] p.ghostSpaceMultiple;
	p.topNeighborGhosts.clear();
	p.topGhosts.clear();
	p.bottomGhosts.clear();
	p.bottomNeighborGhosts.clear();
	for (auto ptr: p.devDims) {
		delete ptr;
	}
	p.devDims.clear();
	Logger::debug("DeviceConfig::deletePlaces: SUCCESS!");
}

int* DeviceConfig::getDimSize() {
	return this->dimSize;
}

int DeviceConfig::getDimensions() {
	return this->dimensions;
}

void DeviceConfig::setDimSize(int *size) {
	this->dimSize = size;
}

void DeviceConfig::setDimensions(int dims) {
	dimensions = dims;
}

std::vector<int> DeviceConfig::getDevices() {
	return activeDevices;
}

void* DeviceConfig::getPlaceStatesForTransfer(int handle, int device) {
	std::vector<std::pair<Place**, void*>> tmp = devPlacesMap[handle].topGhosts;
	return tmp[device].second;
}

void* DeviceConfig::getAgentStatesForTransfer(int handle, int device) {
	return devAgentsMap[handle].devStates[device];
}

__global__ void cleanGhostPointers(Place** p_ptrs, int qty) {
	unsigned idx = getGlobalIdx_1D_1D();
	if (idx < qty) {
		PlaceState* state = (PlaceState*)(p_ptrs[idx]->getState());
		for (int i = 0; i < MAX_NEIGHBORS; ++i) {
			state->neighbors[i] = NULL;
		}
		for (int i = 0; i < MAX_AGENTS; ++i) {
			state->agents[i] = NULL;
		}
	}
}

// copy PlaceState's to neighbor device
void DeviceConfig::copyGhostPlaces(int handle, int stateSize) {
	dim3* pDims = getPlacesThreadBlockDims(handle);
	Logger::debug("DeviceConfig::copyGhostPlaces - entering copyGhostPlaces");
	Logger::debug("Handle: %d and stateSize = %d and num PlaceState top copy: %d", handle, stateSize, MAX_AGENT_TRAVEL * getDimSize()[0]);
    Logger::debug("Number of devices = %d", activeDevices.size());

	cudaEvent_t eventA, eventB;

	#pragma omp parallel 
	{
		int gpu_id = -1;
		CATCH(cudaGetDevice(&gpu_id));
		Logger::debug("copyGhostPlaces: before event recording: device #%d", gpu_id);
		if (gpu_id == 0) {
			cudaEventCreateWithFlags(&eventA, cudaEventDisableTiming);
		} else {
			cudaEventCreateWithFlags(&eventB, cudaEventDisableTiming);
		}
		Logger::debug("copyGhostPlaces: after event recording: device #%d", gpu_id);
		if (gpu_id == 0) {
			// copy from bottom
			CATCH(cudaMemcpy(devPlacesMap[handle].bottomNeighborGhosts.at(gpu_id).second, 
				devPlacesMap[handle].topGhosts.at(gpu_id + 1).second, 
				MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
			cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].bottomNeighborGhosts.at(gpu_id).first, 
				MAX_AGENT_TRAVEL * getDimSize()[0]);
			cudaEventRecord(eventA, 0);
		}

		else if (gpu_id == activeDevices.size() - 1) {
			// copy from top
			CATCH(cudaMemcpy(devPlacesMap[handle].topNeighborGhosts.at(gpu_id).second, 
				devPlacesMap[handle].bottomGhosts.at(gpu_id - 1).second, 
				MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
			cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].topNeighborGhosts.at(gpu_id).first, 
				MAX_AGENT_TRAVEL * getDimSize()[0]);
			cudaEventRecord(eventB, 0);
		}

		else {
			// copy from bottom
			CATCH(cudaMemcpy(devPlacesMap[handle].bottomNeighborGhosts.at(gpu_id).second, 
				devPlacesMap[handle].topGhosts.at(gpu_id + 1).second, 
				MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
			cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].bottomNeighborGhosts.at(gpu_id).first, 
				MAX_AGENT_TRAVEL * getDimSize()[0]);
			// copy from top
			CATCH(cudaMemcpy(devPlacesMap[handle].topNeighborGhosts.at(gpu_id).second, 
				devPlacesMap[handle].bottomGhosts.at(gpu_id - 1).second, 
				MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
			cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].topNeighborGhosts.at(gpu_id).first, 
				MAX_AGENT_TRAVEL * getDimSize()[0]);
		}

		if (gpu_id == 0) {
			cudaEventSynchronize(eventB);
		} else {
			cudaEventSynchronize(eventA);
		}
	}
	Logger::debug("DeviceConfig: exiting copyGhostPlaces");
}

int* DeviceConfig::calculateRandomNumbers(int size, int max_num) {
	// First create an instance of an engine.
    random_device rnd_device;
    // Specify the engine and distribution.
    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    uniform_int_distribution<int> dist {0, max_num - 1};
    
    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };

    vector<int> vec(size);
    generate(begin(vec), end(vec), gen);

	int* randNumArray = &vec[0];
	return randNumArray;
}
} // end Mass namespace
