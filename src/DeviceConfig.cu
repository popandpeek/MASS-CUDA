/**
 *  @file DeviceConfig.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "DeviceConfig.h"
#include "Dispatcher.h"
#include "Agent.h"
#include "Place.h"

namespace mass {

DeviceConfig::DeviceConfig() :
		deviceNum(-1), loaded(false) {
}

DeviceConfig::DeviceConfig(int device) :
		deviceNum(device), loaded(false) {
	cudaSetDevice(deviceNum);
	cudaStreamCreate(&inputStream);
	cudaStreamCreate(&outputStream);
	cudaEventCreate(&deviceEvent);
}

DeviceConfig::~DeviceConfig() {
	cudaSetDevice(deviceNum);
	// destroy streams
	cudaStreamDestroy(inputStream);
	cudaStreamDestroy(outputStream);
	// destroy events
	cudaEventDestroy(deviceEvent);
}

bool DeviceConfig::isLoaded() {
	return loaded;
}
void DeviceConfig::setLoaded(bool loaded) {
	this->loaded = loaded;
}

}
