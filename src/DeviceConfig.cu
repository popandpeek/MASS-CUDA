
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

	for (std::size_t i; i < activeDevices.size(); ++i) {
		CATCH(cudaDeviceReset(activeDevices[i]));
	}

	Logger::debug("Done with deviceConfig freeDevice().");
}

// TODO: Refactor for UVA memory - NO CHANGES NEEDED?
void DeviceConfig::load(void*& destination, const void* source, size_t bytes) {
	CATCH(cudaMemcpy(destination, source, bytes, H2D));
	CATCH(cudaMemGetInfo(&freeMem, &allMem));
}

// TODO: Refactor for UVA memory
void DeviceConfig::unload(void* destination, void* source, size_t bytes) {
	CATCH(cudaMemcpy(destination, source, bytes, D2H));
	CATCH(cudaFree(source));
	CATCH(cudaMemGetInfo(&freeMem, &allMem));
}

int DeviceConfig::countDevPlaces(int handle) {
	if (devPlacesMap.count(handle) != 1) {
		throw MassException("Handle not found.");
	}
	return devPlacesMap[handle].qty;
}

Place** DeviceConfig::getDevPlaces(int handle) {
	return devPlacesMap[handle].devPtr;
}

void* DeviceConfig::getPlaceState(int handle) {
	return devPlacesMap[handle].devState;
}

Agent** DeviceConfig::getDevAgents(int handle) {
	return devAgentsMap[handle].devPtr;
}

void* DeviceConfig::getAgentsState(int handle) {
	return devAgentsMap[handle].devState; 
}

int DeviceConfig::getNumAgents(int handle) {
	return devAgentsMap[handle].nAgents;
}

int DeviceConfig::getNumAgentObjects(int handle) {
	return devAgentsMap[handle].nextIdx;
}

int DeviceConfig::getMaxAgents(int handle) {
	return devAgentsMap[handle].nObjects;
}

int DeviceConfig::getDeviceNum() {
	return deviceNum;
}

dim3* DeviceConfig::getBlockThreadDims(int handle) {
    int numBlocks = (getNumAgentObjects(handle) - 1) / BLOCK_SIZE + 1;
    dim3 blockDim(numBlocks);

    int nThr = (getNumAgentObjects(handle) - 1) / numBlocks + 1;
    dim3 threadDim(nThr);

    devAgentsMap[handle].dims[0] = blockDim;
    devAgentsMap[handle].dims[1] = threadDim;

    return devAgentsMap[handle].dims;
}

__global__ void destroyAgentsKernel(Agent **agents, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		delete agents[idx];
	}
}

void DeviceConfig::deleteAgents(int handle) {
	AgentArray a = devAgentsMap[handle];

	dim3* dims = getBlockThreadDims(handle);
	destroyAgentsKernel<<<dims[0], dims[1]>>>(a.devPtr, a.nObjects);
	CHECK();
	CATCH(cudaFree(a.devPtr));
	CATCH(cudaFree(a.devState));
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

	int blockDim = (p.qty - 1) / BLOCK_SIZE + 1;
	int threadDim = (p.qty - 1) / blockDim + 1;
	destroyPlacesKernel<<<blockDim, threadDim>>>(p.devPtr, p.qty);
	CHECK();
	CATCH(cudaFree(p.devPtr));
	CATCH(cudaFree(p.devState));	
	devPlacesMap.erase(handle);
}

int* DeviceConfig::getSize() {
	return this->size;
}

int DeviceConfig::getDims() {
	return this->dimensions;
}

void DeviceConfig::setSize(int *size) {
	this->size = size;
}

void DeviceConfig::setDims(int dims) {
	dimensions = dims;
}

std::vector<int> DeviceConfig::getDevices() {
	return activeDevices;
}

// void DeviceConfig::setDevices(std::vector<int> devices) {
// 	activeDevices = devices;

// }
} // end Mass namespace
