/**
 *  @file DeviceConfig.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#pragma once

#include <map>

namespace mass {

// forward declarations
class Agent;
class Place;

struct PlaceArray {
	Place** devPtr;
	int qty;
	PlaceArray() {
		devPtr = NULL;
		qty = 0;
	}
};

struct AgentArray {
	Agent** devPtr;
	int qty;
	AgentArray() {
		devPtr = NULL;
		qty = 0;
	}
};

/**
 *  This class represents a computational resource. In most cases it will
 *  represent a GPU, but it could also be used to encapsulate a CPU
 *  computing resource.
 */
class DeviceConfig {
	friend class Dispatcher;

public:
	DeviceConfig();
	DeviceConfig(int device);
	virtual ~DeviceConfig();
 
	void freeDevice();
	void setAsActiveDevice();
  
  // void loadPlaces(PlacesPartition *part);
  // void loadAgents(AgentsPartition *part);
  
	bool isLoaded();
	void setLoaded(bool loaded);

	void setNumPlaces(int numPlaces);

	Place** getPlaces(int rank);
	int getNumPlacePtrs(int rank);

	DeviceConfig( const DeviceConfig& other ); // copy constructor
	DeviceConfig &operator=(const DeviceConfig &rhs); // assignment operator

private:
	int deviceNum;
	cudaStream_t inputStream;
	cudaStream_t outputStream;
	cudaEvent_t deviceEvent;
	PlaceArray devPlaces;
	std::map<int, AgentArray> devAgents;
	bool loaded;

};
// end class

}// end namespace
