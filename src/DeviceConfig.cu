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
#include "Mass.h"

namespace mass {

DeviceConfig::DeviceConfig() :
		deviceNum(-1), loaded(false) {
}

DeviceConfig::DeviceConfig(int device) :
		deviceNum(device), loaded(false) {
	Mass::log("Initializing deviceConfig");
	cudaSetDevice(deviceNum);
	cudaStreamCreate(&inputStream);
	cudaStreamCreate(&outputStream);
	cudaEventCreate(&deviceEvent);
}

DeviceConfig::~DeviceConfig() {
	Mass::log("Destroying deviceConfig");
	cudaSetDevice(deviceNum);
	// destroy streams

	// TODO there is a bug here that crashes the program.
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
