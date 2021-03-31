
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
	freeMem = 0;
	allMem = 0;
	limit = 0;
	randState = NULL;
	int randStateSize = 0;
	Logger::warn("DeviceConfig::NoParam constructor");
}

DeviceConfig::DeviceConfig(std::vector<int> devices) {
	activeDevices = devices;
	devPlacesMap = map<int, PlaceArray>{};
	devAgentsMap = map<int, AgentArray>{};
	for (int i = 0; i < activeDevices.size(); ++i) {
		CATCH(cudaSetDevice(activeDevices.at(i)));
		CATCH(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
		CATCH(cudaMemGetInfo(&freeMem, &allMem));
		Logger::debug("DeviceConfig: Constructor: mem limit == %llu", limit);
		Logger::debug("DeviceConfig: Constructor: allMem == %llu", allMem);
		size_t total =  size_t(2048) * size_t(2048) * size_t(1536);
		CATCH(cudaDeviceSetLimit(cudaLimitMallocHeapSize, total));
		CATCH(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
		Logger::debug("DeviceConfig: Constructor: mem limit == %llu", limit);
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

	devPlacesMap.clear();
	for (std::size_t i = 0; i < activeDevices.size(); ++i) {
		CATCH(cudaSetDevice(activeDevices.at(i)));
		CATCH(cudaDeviceReset());
	}

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

std::vector<std::pair<Place**, void*>> DeviceConfig::getTopGhostPlaces(int handle) {
	return devPlacesMap[handle].topGhosts;
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

int DeviceConfig::getMaxAgents(int handle, int device) {
	return devAgentsMap[handle].maxAgents[device];
}

int* DeviceConfig::getMaxAgents(int handle) {
	return devAgentsMap[handle].maxAgents;
}

void DeviceConfig::setAgentsThreadBlockDims(int handle) {
	// TODO: Need to update aDims to hold multiple sets
	Logger::debug("DeviceConfig::setAgentsMapThreadBlockDims(): numAgents == %d, numDevices == %d", devAgentsMap[handle].nAgents, activeDevices.size());
	
	for (int i = 0; i < activeDevices.size(); ++i) {
		Logger::debug("DeviceConfig::setAgentsMapThreadBlockDims(): maxAgents[%d] = %d", i, devAgentsMap[handle].maxAgents[i]);
		int numBlocks = ((devAgentsMap[handle].maxAgents[i] - 1) / BLOCK_SIZE) + 1;
		int nThr = ((devAgentsMap[handle].maxAgents[i] - 1) / numBlocks) + 1;
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

std::vector<Agent**> DeviceConfig::getBagOAgentsDevPtrs(int agentHandle) {
	return devAgentsMap[agentHandle].collectedAgents;
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
		Logger::debug("DeviceConfig::deleteAgents: device: %d, Number to delete: %d.", i, a.maxAgents[i]);
		cudaSetDevice(activeDevices.at(i));
		destroyAgentsKernel<<<a.aDims.at(i).first, a.aDims.at(i).second>>>(a.devPtrs.at(i), a.maxAgents[i]);
		CHECK();
		cudaDeviceSynchronize();
		CATCH(cudaFree(a.devPtrs.at(i)));
		CATCH(cudaFree(a.devStates.at(i)));
		Logger::debug("DeviceConfig::deleteAgents: CUDA memory freed on device: %d", i);
	}

	delete a.nAgentsDev;
	delete a.maxAgents;
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
    for (int i = 0; i < activeDevices.size(); i+=2) {
        // copy right
        cudaSetDevice(activeDevices.at(i + 1));
		CATCH(cudaMemcpyAsync(devPlacesMap[handle].topNeighborGhosts.at(i + 1).second, 
				devPlacesMap[handle].bottomGhosts.at(i).second, 
				MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
		cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].topNeighborGhosts.at(i + 1).first, MAX_AGENT_TRAVEL * getDimSize()[0]);
        if (i != 0) {
            // copy left
            cudaSetDevice(activeDevices.at(i - 1));
			CATCH(cudaMemcpyAsync(devPlacesMap[handle].bottomNeighborGhosts.at(i - 1).second, 
					devPlacesMap[handle].topGhosts.at(i).second, 
					MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
			cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].bottomNeighborGhosts.at(i - 1).first, MAX_AGENT_TRAVEL * getDimSize()[0]);
        }
    }

    for (int i = 1; i < activeDevices.size(); i+=2) {
        // copy left
        cudaSetDevice(activeDevices.at(i - 1));
		CATCH(cudaMemcpyAsync(devPlacesMap[handle].bottomNeighborGhosts.at(i - 1).second, 
				devPlacesMap[handle].topGhosts.at(i).second, 
				MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
		cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].bottomNeighborGhosts.at(i - 1).first, MAX_AGENT_TRAVEL * getDimSize()[0]);
        if (i != activeDevices.size() - 1) {
            // copy right
            cudaSetDevice(activeDevices.at(i + 1));
			CATCH(cudaMemcpyAsync(devPlacesMap[handle].topNeighborGhosts.at(i + 1).second, 
					devPlacesMap[handle].topGhosts.at(i).second, 
					MAX_AGENT_TRAVEL * getDimSize()[0] * stateSize, cudaMemcpyDefault));
			cleanGhostPointers<<<pDims[0], pDims[1]>>>(devPlacesMap[handle].topNeighborGhosts.at(i + 1).first, MAX_AGENT_TRAVEL * getDimSize()[0]);
        }
    }
}

__global__ void initCurand(curandState *state, int nNums){
    int idx = getGlobalIdx_1D_1D();
	if (idx < nNums) {
		curand_init(clock64(), idx, 0, &state[idx]);
	}
}

__global__ void calculateRandomNumbersKernel(unsigned int* nums, curandState *state, int nNums) {
	int idx = getGlobalIdx_1D_1D();
	if (idx < nNums) {
		nums[idx] = curand_uniform(state);
	}
}

// TODO: Refactor to return device pointer and check for whether the pointer is
//		 on host or device with cudaDeviceGetAttributes()
unsigned int* DeviceConfig::calculateRandomNumbers(int size) {
	if (randStateSize != size) {
		if (randState != NULL) {
			CATCH(cudaFree(randState));
			cudaDeviceSynchronize();
			CHECK();
			randState = NULL;
		}

		randStateSize = size;
		CATCH(cudaMalloc((void**)randState, size * sizeof(curandState)));
		initCurand<<<(size+nTHB-1)/nTHB, nTHB>>>(randState, size);
		cudaDeviceSynchronize();
		CHECK();
	}

	unsigned int *dMem, *hMem;
	hMem = new unsigned int[size*sizeof(unsigned int)];
	CATCH(cudaMalloc(&dMem, size * sizeof(unsigned int)));
	calculateRandomNumbersKernel<<<((size+nTHB-1)/nTHB), nTHB>>>(dMem, randState, size);
	cudaDeviceSynchronize();
	CHECK();
	CATCH(cudaMemcpy(hMem, dMem, size * sizeof(unsigned int), cudaMemcpyDefault));
	CATCH(cudaFree(dMem));
	return hMem;
}

} // end Mass namespace
