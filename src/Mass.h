
#pragma once

#include <iostream>
#include <fstream> // ofstream
#include <stddef.h>
#include <map>

#include "Dispatcher.h"
#include "Places.h"
#include "Agents.h"

namespace mass {

class Mass {
	friend class Places;
	friend class Agents;

public:
	/**
	 *  Initializes the MASS environment using the GPU resource with 
	 *  the highest compute capability discovered on the machine.
	 *  Must be called prior to all other MASS methods.
	 */
	static void init();

	/**
	 *  Shuts down the MASS environment, releasing all resources.
	 */
	static void finish();

	/**
	 *  Gets the places object for this handle.
	 *  @param handle an int that uniquely identifies a places collection.
	 *  @return NULL if not found.
	 */
	static Places *getPlaces(int handle);

	/**
	 *  Gets the number of Places collections in this simulation.
	 */
	static int numPlacesInstances();

	/**
	 * Creates a Places instance with the provided parameters.
	 @param argument is the argument passed to each Places constructor fucntion. 
	  		It should be a void * to a contiguos space of arguments.
	  @param argSize is the size of the contiguous space of arguments specified in argument
	 */
	template<typename P, typename S>
	static Places* createPlaces(int handle, void *argument, int argSize,
			int dimensions, int size[]);

	/**
	 * Creates a Agents instance with the provided parameters.
	  @param argument is the argument passed to each Agents constructor fucntion. 
	  		It should be a void * to a contiguos space of arguments.
	  @param argSize is the size of the contiguous space of arguments specified in argument
	  @param nAgents is the initial number of agents instantiated in the system
	  @param placesHandle is the handle of the Places collection over which to instantiate the Agents collection.
	  @param maxAgents is the maximum number of agents that can be present in the system thoughout the whole simulation.
	  		If the maxAgents parameter is omitted the library set it to the default value of nAgents*2.
	  @param placeIdxs is the array of size nAgents, specifying the index of a place for where each of the Agents 
	  		should be instantiated. If placeIdxs parameter is omitted, the libary randomly distributes agents over the grid of places.
	 */
	template<typename AgentType, typename AgentStateType>
	static Agents* createAgents(int handle, void *argument, int argSize,
			int nAgents, int placesHandle, int maxAgents =0, int* placeIdxs = NULL);


private:

	static std::map<int, Places*> placesMap;
	static std::map<int, Agents*> agentsMap;
	static Dispatcher *dispatcher; /**< The object that handles communication with the GPU(s). */
};

template<typename P, typename S>
Places* Mass::createPlaces(int handle, void *argument, int argSize,
		int dimensions, int size[]) {

	Logger::debug("Entering Mass::createPlaces\n");
	if (dimensions != 2) {
		Logger::warn("The current version of MASS CUDA only supports the 2D dimensionality");
	}
	// create an API object for this Places collection
	Places *places = new Places(handle, dimensions, size, dispatcher);
	placesMap[handle] = places;

	// perform actual instantiation of user classes
	dispatcher->instantiatePlaces<P, S>(handle, argument, argSize, dimensions,
			size, places->numElements);
	Logger::debug("Exiting Mass::createPlaces\n");

	return places;
}

template<typename AgentType, typename AgentStateType>
Agents* Mass::createAgents(int handle, void *argument, int argSize,
		int nAgents, int placesHandle, int maxAgents, int* placeIdxs) {

	Logger::debug("Entering Mass::createAgents\n");

	// create an API object for this Agents collection
	Agents *agents = new Agents(handle, dispatcher, placesHandle);
	agentsMap[handle] = agents;

	// perform actual instantiation of user classes 
	dispatcher->instantiateAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, nAgents, placesHandle, maxAgents, placeIdxs);

	Logger::debug("Exiting Mass::createAgents\n");

	return agents;
}


} /* namespace mass */
