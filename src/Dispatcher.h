

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <queue>

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
	void migrateAgents(int agentHandle, int placeHandle);

	/* Executed during manageAll() call on the collection of agents to complete the 
	 * agent spawning procedure for the agents that invoked the spawn() function.
	 */
	void spawnAgents(int agentHandle);

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
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs);

private:
	bool updateNeighborhood(int handle, std::vector<int*> *vec);
	DeviceConfig *deviceInfo;

	bool initialized;
	bool deviceLoaded;

	std::vector<int*> *neighborhood; /* The previous vector of neighbors.*/

};
// end class

template<typename P, typename S>
void Dispatcher::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty) {
	Logger::debug("Inside Dispatcher::instantiatePlaces\n");

	// Create UMA DataModel
	// modify GPU data model
	deviceInfo->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size,
			qty);
}

template<typename AgentType, typename AgentStateType>
void Dispatcher::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int placesHandle, int maxAgents, int* placeIdxs) {

	Logger::debug("Inside Dispatcher::instantiateAgents\n");

	//create UMA data model
	//create GPU data model
	deviceInfo->instantiateAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, nAgents, placesHandle, maxAgents, placeIdxs);
}

} // namespace mass
