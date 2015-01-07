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

namespace mass {

DeviceConfig::DeviceConfig() :
		deviceNum(-1), loaded(false) {
	Logger::warn("DeviceConfig::NoParam constructor");
}

DeviceConfig::DeviceConfig(int device) :
		deviceNum(device), loaded(false) {
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
	CATCH(cudaSetDevice(deviceNum));

	// TODO there is a bug here that crashes the program.
	// destroy streams
//	CATCH(cudaStreamDestroy(inputStream));
//	CATCH(cudaStreamDestroy(outputStream));
//	// destroy events
//	CATCH(cudaEventDestroy(deviceEvent));
	std::map<int, PlaceArray>::iterator it = devPlacesMap.begin();
	while (it != devPlacesMap.end()) {
		PlaceArray p = it->second;
		deletePlaces(p.devPtr, p.qty);
		CATCH(cudaFree(p.devPtr));
	}
	CATCH(cudaDeviceReset());
	Logger::debug("Done with deviceConfig freeDevice().");
}

bool DeviceConfig::isLoaded() {
	return loaded;
}

void DeviceConfig::setLoaded(bool loaded) {
	this->loaded = loaded;
}

DeviceConfig::DeviceConfig(const DeviceConfig& other) {
	Logger::debug("DeviceConfig copy constructor.");
	deviceNum = other.deviceNum;
//	inputStream = other.inputStream;
//	outputStream = other.outputStream;
//	deviceEvent = other.deviceEvent;
	devPlacesMap = other.devPlacesMap;
	loaded = other.loaded;
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
		loaded = rhs.loaded;
		devAgents = rhs.devAgents;
	}
	return *this;
}

void DeviceConfig::setNumPlaces(int numPlaces) {
	if (numPlaces > devPlacesMap[0].qty) {
		if (NULL != devPlacesMap[0].devPtr) {
			CATCH(cudaFree(devPlacesMap[0].devPtr));
		}
		Place** ptr = NULL;
		CATCH(cudaMalloc((void** ) ptr, numPlaces * sizeof(Place*)));
		devPlacesMap[0].devPtr = ptr;
		devPlacesMap[0].qty = numPlaces;
	}
}

Place** DeviceConfig::getPlaces(int rank) {
	return devPlacesMap[rank].devPtr;
}

int DeviceConfig::getNumPlacePtrs(int rank) {
	return devPlacesMap[0].qty;
}

__global__ void destroyPlacesKernel(Place **places, int qty) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < qty) {
		delete places[idx];
	}
}

void DeviceConfig::deletePlaces(Place **places, int qty) {

	int blockDim = (qty - 1) / BLOCK_SIZE + 1;
	int threadDim = (qty - 1) / blockDim + 1;
	destroyPlacesKernel<<<blockDim, threadDim>>>(places, qty);
	CHECK();
}

} // end Mass namespace
