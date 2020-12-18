
#include <curand.h>

#include "DeviceConfig.h"
#include "Place.h"
#include "cudaUtil.h"
#include "Logger.h"
#include "MassException.h"
#include "string.h"

using namespace std;

namespace mass {

DeviceConfig::DeviceConfig() :
		activeDevices(-1) {
	freeMem = 0;
	allMem = 0;
	Logger::warn("DeviceConfig::NoParam constructor");
}

DeviceConfig::DeviceConfig(std::vector<int> devices) {
	activeDevices = devices;
    devPlacesMap = map<int, PlaceArray>{};
	devAgentsMap = map<int, AgentArray>{};
}

DeviceConfig::~DeviceConfig() {
	Logger::debug("deviceConfig destructor ");
}

void DeviceConfig::freeDevice() {
	Logger::debug("deviceConfig free ");

	// Delete agents:
	std::map<int, AgentArray>::iterator it_a = devAgentsMap.begin();
	while (it_a != devAgentsMap.end()) {
		deleteAgents(it_a->first);
		++it_a;
	}
	devAgentsMap.clear();

	// Delete places:
	std::map<int, PlaceArray>::iterator it_p = devPlacesMap.begin();
	while (it_p != devPlacesMap.end()) {
		deletePlaces(it_p->first);
		++it_p;
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
	int numBlocks = (devPlacesMap[handle].placesStride + (2 * dimSize[0] * devPlacesMap[handle].ghostSpaceMultiple[0])) / BLOCK_SIZE + 1;
	int nThr = (devPlacesMap[handle].placesStride + (2 * dimSize[0] * devPlacesMap[handle].ghostSpaceMultiple[0])) / numBlocks + 1;
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
	Logger::debug("DeviceConfig::setAgentsMapThreadBlockDims(): numPlaces == %d, numDevices == %d", devAgentsMap[handle].nAgents, activeDevices.size());
	int numBlocks = (devAgentsMap[handle].nAgents / activeDevices.size()) / BLOCK_SIZE + 1;
	int nThr = (devAgentsMap[handle].nAgents / activeDevices.size()) / numBlocks + 1;
	dim3 bDim = dim3(numBlocks);
	dim3 tDim = dim3(nThr);

	devAgentsMap[handle].aDims[0] = bDim;
	devAgentsMap[handle].aDims[1] = tDim;
	Logger::debug("setAgentsThreadBlockDims(): numBlocks == %u, %u, %u; nThr == %u, %u, %u", bDim.x, bDim.y, bDim.z, tDim.x, tDim.y, tDim.z);
}

int* DeviceConfig::getnAgentsDev(int handle) {
	return devAgentsMap[handle].nAgentsDev;
}

std::vector<Agent**> DeviceConfig::getBagOAgentsDevPtrs(int agentHandle) {
	return devAgentsMap[agentHandle].collectedAgents;
}

dim3* DeviceConfig::getAgentsThreadBlockDims(int handle) {
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
	AgentArray a = devAgentsMap[handle];
	dim3* aDims = getAgentsThreadBlockDims(handle);

	for (int i = 0; i < a.devPtrs.size(); ++i) {
		destroyAgentsKernel<<<a.aDims[0], a.aDims[1]>>>(a.devPtrs.at(i), a.nAgentsDev[i]);
		CHECK();
		CATCH(cudaFree(a.devPtrs.at(i)));
		CATCH(cudaFree(a.devStates.at(i)));
		cudaDeviceSynchronize();
	}

	devAgentsMap.erase(handle);
}

__global__ void destroyPlacesKernel(Place **places, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		delete places[idx];
	}
}

void DeviceConfig::deletePlaces(int handle) {
	PlaceArray p = devPlacesMap[handle];
	dim3* pDims = getPlacesThreadBlockDims(handle);

	
	for (int i = 0; i < p.devPtrs.size(); ++i) {
		destroyPlacesKernel<<<p.pDims[0], p.pDims[1]>>>(p.devPtrs.at(i), p.placesStride);
		CHECK();
		CATCH(cudaFree(p.devPtrs.at(i)));
		CATCH(cudaFree(p.devStates.at(i)));
	}

	devPlacesMap.erase(handle);
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

} // end Mass namespace
