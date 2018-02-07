
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
		deviceNum(-1) {
	freeMem = 0;
	allMem = 0;
	Logger::warn("DeviceConfig::NoParam constructor");
}

DeviceConfig::DeviceConfig(int device) :
		deviceNum(device) {
	Logger::debug("DeviceConfig(int) constructor");
	CATCH(cudaSetDevice(deviceNum));
	CATCH(cudaMemGetInfo(&freeMem, &allMem));
	CATCH(cudaDeviceSetLimit(cudaLimitMallocHeapSize, allMem * 3 / 4));
	devPlacesMap = map<int, PlaceArray>{};
	devAgentsMap = map<int, AgentArray>{};
}

DeviceConfig::~DeviceConfig() {
	Logger::debug("deviceConfig destructor ");
}

void DeviceConfig::freeDevice() {
	Logger::debug("deviceConfig free ");

	std::map<int, PlaceArray>::iterator it = devPlacesMap.begin();
	while (it != devPlacesMap.end()) {
		deletePlaces(it->first);
		++it;
	}
	devPlacesMap.clear();

	CATCH(cudaDeviceReset());
	Logger::debug("Done with deviceConfig freeDevice().");
}

void DeviceConfig::load(void*& destination, const void* source, size_t bytes) {
	CATCH(cudaMalloc((void** ) &destination, bytes));
	CATCH(cudaMemcpy(destination, source, bytes, H2D));
	CATCH(cudaMemGetInfo(&freeMem, &allMem));
}

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

int DeviceConfig::getNumPlacePtrs(int handle) {
	return devPlacesMap[handle].qty;
}

Agent** DeviceConfig::getDevAgents(int handle) {
	return devAgentsMap[handle].devPtr;
}

void* DeviceConfig::getAgentsState(int handle) {
	return devAgentsMap[handle].devState; 
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

} // end Mass namespace
