/**
 *  @file Dispatcher.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <queue>

#include "DeviceConfig.h"
#include "DataModel.h"

namespace mass {

// forward declarations
class AgentsPartition;
class PlacesPartition;

class Dispatcher {

public:

	/**
	 *  Is the Dispatcher constructor.
	 *  The Dispatcher must be initialized prior to use.
	 */
	Dispatcher();

	/**
	 *  Is the Dispatcher initializer.
	 *  the number of GPUs is passed to the initializer. The Dispatcher
	 *  then locates the GPUs, sets up communication links, and prepares to begin
	 *  dispatching data to and from the GPU.
	 *
	 *  @param ngpu the number of GPUs to use in this simulation. 0 if all GPU resources are to be used.
	 *  @param models the data model for this simulation
	 */
	void init(int &ngpu);

	~Dispatcher();

	/**************************************************************************
	 * These are the commands that will execute commands on places objects on
	 * the GPU.
	 **************************************************************************/

	/**
	 * Called when the user wants to look at the data model on the host. This
	 * will extract the most current data from the GPU for the specified places
	 * collection.
	 * @param handle the handle of the places object to refresh.
	 */
	Place** refreshPlaces(int placeHandle);

	/**
	 * Call the specified function on the specified places object with the given
	 * argument.
	 * @param handle the handle of the places upon which to call all
	 * @param functionId the ID of the user defined function
	 * @param argument the function argument. Can be NULL.
	 * @param argSize the size of argument in bytes. If argument == NULL, argSize is not used.
	 */
	void callAllPlaces(int placeHandle, int functionId, void *argument,
			int argSize);

	/**
	 * Call the specified function on the specified places object with the given
	 * argument
	 * @param handle the handle of the places upon which to call all
	 * @param functionId the ID of the user defined function
	 * @param arguments an array of arguments, 1 per place
	 * @param argSize the size of argument in bytes. If argument == NULL, argSize is not used.
	 * @param retSize the size of the return value in bytes. If retSize == 0, NULL will be returned.
	 * @return an array with one element of argSize per place. NULL if retSize == 0
	 */
	void *callAllPlaces(int placeHandle, int functionId, void *arguments[],
			int argSize, int retSize);

	/**
	 *  This function causes all Place elements to call the function specified on all neighboring
	 *  place elements. The offsets to neighbors are defined in the destinations vector (a collection
	 *  of offsets from the caller to the callee place elements). The caller cell's outMessage is a
	 *  continuous set of arguments passed to the callee's method. The caller's inMessages[] stores
	 *  values returned from all callees. More specifically, inMessages[i] maintains a set of return
	 *  from the ith neighbor in destinations.
	 *  Example destinations vector:
	 *    vector<int*> destinations;
	 *    int north[2] = {0, 1}; destinations.push_back( north );
	 *    int east[2] = {1, 0}; destinations.push_back( east );
	 *    int south[2] = {0, -1}; destinations.push_back( south );
	 *    int west[2] = {-1, 0}; destinations.push_back( west );
	 */
	void exchangeAllPlaces(int placeHandle, int functionId,
			std::vector<int*> *destinations);

	/**
	 *  Exchanges the boundary places with the left and right neighboring nodes.
	 *  @param handle the handle for which boundaries should be exchanged
	 */
	void exchangeBoundaryPlaces(int placeHandle);

	/**************************************************************************
	 * These are the commands that will execute commands on agents objects on
	 * the GPU.
	 **************************************************************************/

	/**
	 * Called when the user wants to look at the data model on the host. This
	 * will extract the most current data from the GPU for the specified agents
	 * collection.
	 * @param agents the agents object to refresh.
	 */
	Agent** refreshAgents(int handle);

	/**
	 * Calls the specified function on the specified agents group with argument
	 * as the function parameter.
	 * @param agents the agents object to call all on
	 * @param functionId the user specified function ID
	 * @param argument the argument for the function
	 * @param argSize the size in bytes of the argument
	 */
	void callAllAgents(int handle, int functionId, void *argument, int argSize);

	/**
	 * Calls the specified function on the specified agents group with argument
	 * as the function parameter. Returns a void* with an element of retSize for
	 * every agent in the group.
	 *
	 * @param agents the agents object to call all on
	 * @param functionId the user specified function ID
	 * @param arguments one void* argument per agent
	 * @param argSize the size in bytes of each argument
	 * @param retSize the size in bytes of the return value
	 * @return a void* with an element of retSize for every agent in the group.
	 */
	void *callAllAgents(int handle, int functionId, void *arguments[],
			int argSize, int retSize);

	/**
	 * Calls manage on all agents of the specified group.
	 * @param agents the agents to manage.
	 */
	void manageAllAgents(int handle);

	template<typename P, typename S>
	void instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty, int boundary_width);

private:

	DeviceConfig *getNextDevice();
	void unloadDevice(DeviceConfig *device);

	std::map<Partition*, DeviceConfig*> partToDevice;
	std::map<DeviceConfig*, Partition*> deviceToPart;
	std::vector<DeviceConfig> deviceInfo;

	int nextDevice; // tracks which device in deviceInfo is next to be used
	DataModel *model;
	bool initialized;

};
// end class

template<typename P, typename S>
void Dispatcher::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty, int boundary_width) {

	// modify host-side data model
	model->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size,
			qty, boundary_width);

	// TODO this does not yet handle multiple places reliably
	// TODO this does not yet handle more partitions than GPUs
	for (int i = 0; i < deviceInfo.size(); ++i) {
		DeviceConfig *d = getNextDevice();
		Partition *p = model->getPartition(i);

		int objCount = p->getPlacesPartition(handle)->sizeWithGhosts();
		d->instantiatePlaces<P, S>(handle, argument, argSize, objCount);
		partToDevice[p] = d;
		deviceToPart[d] = p;
	}
}

} // namespace mass
