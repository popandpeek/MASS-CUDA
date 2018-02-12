
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <queue>

#include "DataModel.h"
#include "DeviceConfig.h"


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
	// void *callAllPlaces(int placeHandle, int functionId, void *arguments[],
	// 		int argSize, int retSize);

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
	void exchangeAllPlaces(int placeHandle, std::vector<int*> *destinations);
	
	void exchangeAllPlaces(int handle, std::vector<int*> *destinations, int functionId, 
		void *argument, int argSize);

	//TODO: add descriptions to all functions
	void callAllAgents(int agentHandle, int functionId, void *argument,
			int argSize);

	// void *callAllAgents(int agentHandle, int functionId, void *arguments[],
	// 		int argSize, int retSize);

	void terminateAgents(int agentHandle);

	void migrateAgents(int agentHandle, int placeHandle);

	void spawnAgents(int agentHandle, int placeHandle);

	int getNumAgents(int agentHandle);
	int getNumAgentObjects(int agentHandle);

	/**
	 * Called when the user wants to look at the data model on the host. This
	 * will extract the most current data from the GPU for the specified agents
	 * collection. It will also update the numAgents reference with the current agent count.
	 */
	Agent** refreshAgents(int agentHandle);


	template<typename P, typename S>
	void instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	void instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle);

private:
	bool updateNeighborhood(int handle, std::vector<int*> *vec);
	DeviceConfig* deviceInfo;

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

	// modify host-side data model
	model->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size,
			qty);
	
	//modify GPU data model
	deviceInfo->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size,
			qty);
}

template<typename AgentType, typename AgentStateType>
void Dispatcher::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle) {

	Logger::debug("Inside Dispatcher::instantiateAgents\n");

	//create GPU data model
	deviceInfo->instantiateAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, nAgents, placesHandle);

	//create host-side data model
	model->instantiateAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, deviceInfo->getMaxAgents(handle));

}

} // namespace mass
