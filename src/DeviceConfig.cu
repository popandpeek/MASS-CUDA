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
	Mass::logger.debug("Initializing deviceConfig");
	CATCH(cudaSetDevice(deviceNum));
	CATCH(cudaStreamCreate(&inputStream));
	CATCH(cudaStreamCreate(&outputStream));
	CATCH(cudaEventCreate(&deviceEvent));
}

DeviceConfig::~DeviceConfig() {
	Mass::logger.debug("deviceConfig destructor ");
}

void DeviceConfig::setAsActiveDevice(){
	Mass::logger.debug("Active device is now %d", deviceNum);
	CATCH(cudaSetDevice(deviceNum));
}

void DeviceConfig::freeDevice() {
	Mass::logger.debug("deviceConfig free ");
	CATCH(cudaSetDevice(deviceNum));

	// TODO there is a bug here that crashes the program.
	// destroy streams
	CATCH(cudaStreamDestroy(inputStream));
	CATCH(cudaStreamDestroy(outputStream));
	// destroy events
	CATCH(cudaEventDestroy(deviceEvent));
	Mass::logger.debug("Done with deviceConfig freeDevice().");
}

bool DeviceConfig::isLoaded() {
	return loaded;
}

void DeviceConfig::setLoaded(bool loaded) {
	this->loaded = loaded;
}

DeviceConfig::DeviceConfig(const DeviceConfig& other) {
	Mass::logger.debug("DeviceConfig copy constructor.");
	deviceNum = other.deviceNum;
	inputStream = other.inputStream;
	outputStream = other.outputStream;
	deviceEvent = other.deviceEvent;
	devPlaces = other.devPlaces;
	loaded = other.loaded;
	devAgents = other.devAgents;
}

DeviceConfig &DeviceConfig::operator=(const DeviceConfig &rhs) {
	Mass::logger.debug("DeviceConfig assignment operator.");
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

void DeviceConfig::setNumPlaces(int numPlaces){
	if(numPlaces > devPlaces.qty){
		if(NULL != devPlaces.devPtr){
			CATCH(cudaFree(devPlaces.devPtr));
		}
		Place** ptr = NULL;
		CATCH(cudaMalloc((void**) ptr, numPlaces * sizeof(Place*)));
		devPlaces.devPtr = ptr;
		devPlaces.qty = numPlaces;
	}
}


Place** DeviceConfig::getPlaces(int rank){
	return devPlaces.devPtr;
}


int DeviceConfig::getNumPlacePtrs(int rank){
	return devPlaces.qty;
}

} // end Mass namespace
