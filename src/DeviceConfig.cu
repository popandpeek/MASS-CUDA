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
#include "cudaUtil.h"

namespace mass {

DeviceConfig::DeviceConfig() :
		deviceNum(-1), loaded(false) {
}

DeviceConfig::DeviceConfig(int device) :
		deviceNum(device), loaded(false) {
	Mass::log("Initializing deviceConfig");
	CATCH(cudaSetDevice(deviceNum));
	CATCH(cudaStreamCreate(&inputStream));
	CATCH(cudaStreamCreate(&outputStream));
	CATCH(cudaEventCreate(&deviceEvent));
}

DeviceConfig::~DeviceConfig() {
	Mass::log("deviceConfig destructor ");
}

void DeviceConfig::free() {
	Mass::log("deviceConfig free ");
	CATCH(cudaSetDevice(deviceNum));
	// destroy streams

	// TODO there is a bug here that crashes the program.
	CATCH(cudaStreamDestroy(inputStream));
	CATCH(cudaStreamDestroy(outputStream));
	// destroy events
	CATCH(cudaEventDestroy(deviceEvent));
}

bool DeviceConfig::isLoaded() {
	return loaded;
}

void DeviceConfig::setLoaded(bool loaded) {
	this->loaded = loaded;
}

DeviceConfig::DeviceConfig(const DeviceConfig& other) {
	Mass::log("DeviceConfig copy constructor.");
	deviceNum = other.deviceNum;
	inputStream = other.inputStream;
	outputStream = other.outputStream;
	deviceEvent = other.deviceEvent;
	devPlaces = other.devPlaces;
	loaded = other.loaded;
	devAgents = other.devAgents;
}

DeviceConfig &DeviceConfig::operator=(const DeviceConfig &rhs) {
	Mass::log("DeviceConfig assignment operator.");
	if (this != &rhs) {

		deviceNum = rhs.deviceNum;
		inputStream = rhs.inputStream;
		outputStream = rhs.outputStream;
		deviceEvent = rhs.deviceEvent;
		devPlaces = rhs.devPlaces;
		loaded = rhs.loaded;
		devAgents = rhs.devAgents;
	}
	return *this;
}
} // end Mass namespace
