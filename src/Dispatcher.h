#ifndef DISPATCHER_H
#define DISPATCHER_H

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <queue>

#include "DeviceConfig.h"
#include "DataModel.h"

namespace mass {

class Dispatcher {

public:

	/**
	 *  Is the Dispatcher constructor.
	 *  The Dispatcher must be initialized prior to use.
	 */
	Dispatcher();

	/**
	 *  Is the Dispatcher initializer.
	 *  The Dispatcher locates the GPU, sets up communication links, and prepares to begin
	 *  dispatching data to and from the GPU.
	 */
	void init();

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
	std::vector<Place**> refreshPlaces(int placeHandle);

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
	 *  This function causes all Place elements to exchange information about their neighbors.
	 *  The neighbors array in each of the places is populated with pointers to the Places in specified 
	 *  in the destinations vector. The offsets to neighbors are defined in the destinations vector (a collection
	 *  of offsets from the caller to the callee place elements).
	 *  Example destinations vector:
	 *    vector<int*> destinations;
	 *    int north[2] = {0, 1}; destinations.push_back( north );
	 *    int east[2] = {1, 0}; destinations.push_back( east );
	 *    int south[2] = {0, -1}; destinations.push_back( south );
	 *    int west[2] = {-1, 0}; destinations.push_back( west );
	 */
	void exchangeAllPlaces(int placeHandle, std::vector<int*> *destinations);
	
	/**
	 *  This function causes all Place elements to exchange information about their neighbors and to call 
	 *  the function specified with functionId on each of the places afterwards.
	 *  In addition to the fuctionality of the standard exchangeAllPlaces function specified above 
	 *  it also takes functionId as a parameter and arguments to that functiom. 
	 *  When the data is collected from the neighboring places, 
	 *  the specified function is executed on all of the places with specified parameters.
	 *  The rationale behind implemening this version of exchangeAllPlaces is performance optimization:
	 *  the data cached during data collection step can be used for the data calculation and thus minimize
	 *  the number of memeory fetches and improve performance.
	 */
	void exchangeAllPlaces(int handle, std::vector<int*> *destinations, int functionId, 
		void *argument, int argSize);

	/**
	 * Call the specified function on the specified places object with the given
	 * argument.
	 * @param handle the handle of the places upon which to call all
	 * @param functionId the ID of the user defined function
	 * @param argument the function argument. Can be NULL.
	 * @param argSize the size of argument in bytes. If argument == NULL, argSize is not used.
	 */
	void callAllAgents(int agentHandle, int functionId, void *argument,
			int argSize);

	/* Executed during manageAll() call on the collection of agents to complete the 
	 * deallocation/memory management of the agents marked for termination.
	 */
	void terminateAgents(int agentHandle);

	/* Executed during manageAll() call on the collection of agents to complete the 
	 * migration of the agents marked for migration.
	 */
	// template<typename AgentType, typename AgentStateType>
	// void migrateAgents(int agentHandle, int placeHandle);

	/* Executed during manageAll() call on the collection of agents to complete the 
	 * agent spawning procedure for the agents that invoked the spawn() function.
	 */
	void spawnAgents(int agentHandle);

	int getNumAgents(int agentHandle);

	int* getMaxAgents(int agentHandle);
	
	int* getNumAgentsInstantiated(int handle);

	int getAgentStateSize(int handle);

	int* getNAgentsDev(int handle);

	int getNumPlaces(int handle);

	int getPlacesStride(int handle);

	int* getGhostPlaceMultiples(int handle);

	/**
	 * Called when the user wants to look at the data model on the host. This
	 * will extract the most current data from the GPU for the specified agents
	 * collection. It will also update the numAgents reference with the current agent count.
	 */
	std::vector<Agent**> refreshAgents(int agentHandle);


	template<typename P, typename S>
	void instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	void instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs);

	template<typename AgentType, typename AgentStateType>
	void manageAll(int agentHandle, int placeHandle);

private:
	bool updateNeighborhood(int handle, std::vector<int*> *vec);
	DeviceConfig *deviceInfo;
	DataModel *model;
	bool initialized;
	bool deviceLoaded;

	std::vector<int*> *neighborhood; /* The previous vector of neighbors.*/

};
// end class

template<typename P, typename S>
void Dispatcher::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty) {
	Logger::debug("Inside Dispatcher::instantiatePlaces\n");

	// create host-side data model
	model->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size,
			qty);

	Logger::debug("Dispatcher::instantiatePlaces: after host model - placesSize[0] = %d, placesSize[1] = %d", size[0], size[1]);

	// create GPU data model
	deviceInfo->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size,
			qty);

	Logger::debug("Dispatcher::instantiatePlaces: after device model - placesSize[0] = %d, placesSize[1] = %d", size[0], size[1]);
}

template<typename AgentType, typename AgentStateType>
void Dispatcher::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs) {

	Logger::debug("Inside Dispatcher::instantiateAgents\n");

	//create GPU data model
	deviceInfo->instantiateAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, nAgents, placesHandle, maxAgents, placeIdxs);

	//create host-side data model
	model->instantiateAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, nAgents, deviceInfo->getMaxAgents(handle), deviceInfo->getnAgentsDev(handle));
	
	Logger::debug("Exiting Dispatcher::instantiateAgents\n");
}

template<typename AgentType, typename AgentStateType>
void Dispatcher::manageAll(int agentHandle, int placeHandle) {
    // Step 1: kill all agents that need killing
    // terminateAgents(handle);

    // Step 2: migrate all agents that need migrating
    deviceInfo->migrateAgents<AgentType, AgentStateType>(agentHandle, placeHandle);

    // Step 3: spawn all new agents that need spawning
    // spawnAgents(handle);
}

} // namespace mass
#endif
