/**
 *  @file DeviceConfig.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "DeviceConfig.h"
#include "Agent.h"
#include "Place.h"
#include "cudaUtil.h"
#include "Logger.h"

using namespace std;

namespace mass {

DeviceConfig::DeviceConfig() :
		deviceNum(-1) {
	Logger::warn("DeviceConfig::NoParam constructor");
}

DeviceConfig::DeviceConfig(int device) :
		deviceNum(device) {
	Logger::debug("DeviceConfig(int) constructor");
	CATCH(cudaSetDevice(deviceNum));
//	CATCH(cudaStreamCreate(&inputStream));
//	CATCH(cudaStreamCreate(&outputStream));
//	CATCH(cudaEventCreate(&deviceEvent));
}

DeviceConfig::~DeviceConfig() {
	Logger::debug("deviceConfig destructor ");
}

void DeviceConfig::setAsActiveDevice() {
	Logger::debug("Active device is now %d", deviceNum);
	CATCH(cudaSetDevice(deviceNum));
}

void DeviceConfig::freeDevice() {
	Logger::debug("deviceConfig free ");
	setAsActiveDevice();

	// TODO there is a bug here that crashes the program.
	// destroy streams
//	CATCH(cudaStreamDestroy(inputStream));
//	CATCH(cudaStreamDestroy(outputStream));
//	// destroy events
//	CATCH(cudaEventDestroy(deviceEvent));

	std::map<int, PlaceArray>::iterator it = devPlacesMap.begin();
	while (it != devPlacesMap.end()) {
		deletePlaces(it->first);
		++it;
	}
	devPlacesMap.clear();
	CATCH(cudaDeviceReset());

	Logger::debug("Done with deviceConfig freeDevice().");
}

void DeviceConfig::loadPartition(Partition* partition, int placeHandle) {
	map<int, PlacesPartition*> parts = partition->getPlacesPartitions();
	PlacesPartition *pPart = parts[placeHandle];

	void*& dest = devPlacesMap[placeHandle].devState;
	void* src = ((Place*) pPart->getLeftGhost())->getState();
	size_t sz = pPart->getPlaceBytes() * pPart->sizeWithGhosts();
	if (NULL == dest) {
		CATCH(cudaMalloc((void** ) &dest, sz));
	}
	CATCH(cudaMemcpy(dest, src, sz, H2D));

//	// load all corresponding agents partitions of the same rank
//	map<int, AgentsPartition*> agents = partition->getAgentsPartitions(placeHandle);
//
//	map<int, AgentsPartition*>::iterator it = agents.begin();
//	while (it != agents.end()) {
//		AgentsPartition *aPart = it->second;
//		if (!aPart->isLoaded()) {
//			Logger::debug("Loading agents rank %d", handle);
//			d->loadAgentsPartition(aPart);
//			loadedAgents[aPart] = d;
//		}
//		++it;
//	}
}

void DeviceConfig::load(void*& destination, const void* source, size_t bytes) {
	CATCH(cudaMalloc((void** ) &destination, bytes));
	CATCH(cudaMemcpy(destination, source, bytes, H2D));
}

void DeviceConfig::unload(void* destination, void* source, size_t bytes) {
	CATCH(cudaMemcpy(destination, source, bytes, D2H));
	CATCH(cudaFree(source));
}

DeviceConfig::DeviceConfig(const DeviceConfig& other) {
	Logger::debug("DeviceConfig copy constructor.");
	deviceNum = other.deviceNum;
//	inputStream = other.inputStream;
//	outputStream = other.outputStream;
//	deviceEvent = other.deviceEvent;
	devPlacesMap = other.devPlacesMap;
	devAgents = other.devAgents;
}

DeviceConfig &DeviceConfig::operator=(const DeviceConfig &rhs) {
	Logger::debug("DeviceConfig assignment operator.");
	if (this != &rhs) {

		deviceNum = rhs.deviceNum;
//		inputStream = rhs.inputStream;
//		outputStream = rhs.outputStream;
//		deviceEvent = rhs.deviceEvent;
		devPlacesMap = rhs.devPlacesMap;
		devAgents = rhs.devAgents;
	}
	return *this;
}

Place** DeviceConfig::getDevPlaces(int handle) {
	return devPlacesMap[handle].devPtr;
}

void* DeviceConfig::getPlaceState(int handle){
	return devPlacesMap[handle].devState;
}
void* DeviceConfig::getAgentState(int handle){
	return devAgents[handle].devState;
}

int DeviceConfig::getNumPlacePtrs(int handle) {
	return devPlacesMap[handle].qty;
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
