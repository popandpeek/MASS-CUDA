/**
 *  @file Dispatcher.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef DISPATCHER_H_
#define DISPATCHER_H_
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include "Agent.h"
#include "Agents.h"
#include "Place.h"
#include "Places.h"
namespace mass {

// forward declarations
class Agents;
class Command;
class Places;

struct DeviceData {
    int deviceNum;
    cudaStream_t inputStream;
    cudaStream_t outputStream;
    cudaEvent_t deviceEvent;
};

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
	void init(int ngpu, Model *model);

	~Dispatcher();

	/**************************************************************************
	 * These are the commands that will execute commands on places objects on
	 * the GPU.
	 **************************************************************************/

	/**
	 *  Creates a Places object.
	 *
	 *  @param handle the unique identifier of this places collections
	 *  @param argument a continuous space of arguments used to initialize the places
	 *  @param argSize the size in bytes of the argument array
	 *  @param dimensions the number of dimensions in the places matrix (i.e. is it 1D, 2D, 3d?)
	 *  @param size the size of each dimension. This MUST be dimensions elements long.
	 *  @return the created Places collection
	 */
	template<typename T>
	Places *createPlaces(int handle, void *argument, int argSize,
			int dimensions, int size[]) {
		// TODO implement
		return NULL;
	}

	/**
	 * Called when the user wants to look at the data model on the host. This
	 * will extract the most current data from the GPU for the specified places
	 * collection.
	 * @param handle the handle of the places object to refresh.
	 */
	void refreshPlaces(Places *places);

	/**
	 * Call the specified function on the specified places object with the given
	 * argument.
	 * @param handle the handle of the places upon which to call all
	 * @param functionId the ID of the user defined function
	 * @param argument the function argument. Can be NULL.
	 * @param argSize the size of argument in bytes. If argument == NULL, argSize is not used.
	 */
	void callAllPlaces(Places *places, int functionId, void *argument, int argSize);

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
	void *callAllPlaces(Places *places, int functionId, void *arguments[],
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
	void exchangeAllPlaces(Places *places, int functionId,
			std::vector<int*> *destinations);

	/**
	 *  Exchanges the boundary places with the left and right neighboring nodes.
	 *  @param handle the handle for which boundaries should be exchanged
	 */
	void exchangeBoundaryPlaces(Places *places);

	/**************************************************************************
	 * These are the commands that will execute commands on agents objects on
	 * the GPU.
	 **************************************************************************/

	/**
	 * Creates an Agents object with the specified parameters.
	 * @param handle the unique number that will identify this Agents object.
	 * @param argument an argument that will be passed to all Agents upon creation
	 * @param argSize the size in bytes of argument
	 * @param places the Places object upon which this agents collection will operate.
	 * @param initPopulation the starting number of agents to instantiate
	 * @return the created Agents collection
	 */
	template<typename T>
	Agents *createAgents(int handle, void *argument, int argSize,
			Places *places, int initPopulation) {
		// TODO implement
		return NULL;
	}

	/**
	 * Called when the user wants to look at the data model on the host. This
	 * will extract the most current data from the GPU for the specified agents
	 * collection.
	 * @param handle the handle of the agents object to refresh.
	 */
    void refreshAgents ( Agents *agents );

	/**
	 * Calls the specified function on the specified agents group with argument
	 * as the function parameter.
	 * @param handle the handle of the agents object to call all on
	 * @param functionId the user specified function ID
	 * @param argument the argument for the function
	 * @param argSize the size in bytes of the argument
	 */
    void callAllAgents ( Agents *agents, int functionId, void *argument, int argSize );

	/**
	 * Calls the specified function on the specified agents group with argument
	 * as the function parameter. Returns a void* with an element of retSize for
	 * every agent in the group.
	 *
	 * @param handle the handle of the agents object to call all on
	 * @param functionId the user specified function ID
	 * @param arguments one void* argument per agent
	 * @param argSize the size in bytes of each argument
	 * @param retSize the size in bytes of the return value
	 * @return a void* with an element of retSize for every agent in the group.
	 */
    void *callAllAgents ( Agents *agents, int functionId, void *arguments[ ],
			int argSize, int retSize);

	/**
	 * Calls manage on all agents of the specified group.
	 * @param handle the handle of the agents to manage.
	 */
    void manageAllAgents ( Agents *agents );

private:

    void loadPlacesPartition ( PlacesPartition *part, DeviceData d );
    void getPlacesPartition ( PlacesPartition *part, bool freeOnRetrieve = true );

    void loadAgentsPartition ( AgentsPartition *part, DeviceData d );
    void getAgentsPartition ( AgentsPartition *part, bool freeOnRetrieve = true );


    std::map<PlacesPartition *, DeviceData> loadedPlaces; // tracks which partition is loaded on which GPU
    std::map<AgentsPartition*, DeviceData> loadedAgents; // tracks whicn partition is loaded on which GPU
    std::queue<DeviceData> deviceInfo;
};
// end class
}// namespace mass

#endif
