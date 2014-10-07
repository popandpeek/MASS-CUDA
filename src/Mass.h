/**
 *  @file Mass.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <iostream>
#include <stddef.h>
#include <map>

namespace mass {

class Agents;
class Dispatcher;
class Places;

class Mass {
    friend class Agents;
    friend class Places;

public:

	/**
	 *  Initializes the MASS environment. Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 *  @param ngpu the number of GPUs to use
	 */
	static void init(std::string args[], int ngpu);

	/**
	 *  Initializes the MASS environment using all available GPU resources. 
	 *  Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 */
	static void init(std::string args[]);

	/**
	 *  Shuts down the MASS environment, releasing all resources.
	 */
	static void finish();

	/**
	 *  Gets the places object for this handle.
	 *  @param handle an int that corresponds to a places object.
	 *  @return NULL if not found.
	 */
    static Places *getPlaces ( int handle );

    /**
    *  Gets the number of Places collections in this simulation.
    */
    static int numPlacesInstances ( );

	/**
	 *  Gets the agents object for this handle.
	 *  @param handle an int that corresponds to an agents object.
	 *  @return NULL if not found.
	 */
	static Agents *getAgents(int handle);

    /**
     *  Gets the number of Agents collections in this simulation.
     */
    static int numAgentsInstances ( );


	/**
	 * Creates a Places object with the specified parameters.
	 * @param handle the unique number that will identify this Places object.
	 * @param classname the name of the Place class to be dynamically loaded
	 * @param argument an argument that will be passed to all Places upon creation
	 * @param argSize the size in bytes of argument
	 * @param dimensions the number of dimensions in the places matrix
	 * (i.e. 1, 2, 3, etc...)
	 * @param size the size of each dimension ordered {numRows, numCols,
	 * numDeep, ...}. This must be dimensions elements long.
	 * @return a pointer the the created places object
	 */
	static Places *createPlaces(int handle, std::string classname, void *argument,
			int argSize, int dimensions, int size[], int boundary_width);

	/**
	 * Creates an Agents object with the specified parameters.
	 * @param handle the unique number that will identify this Agents object.
	 * @param classname the name of the Agent class to be dynamically loaded
	 * @param argument an argument that will be passed to all Agents upon creation
	 * @param argSize the size in bytes of argument
	 * @param places the Places object upon which this agents collection will operate.
	 * @param initPopulation the starting number of agents to instantiate
	 * @return
	 */
	static Agents *createAgents(int handle, std::string classname, void *argument, int argSize,
			Places *places, int initPopulation);

private:
    static std::map<int, Places*> placesMap;
    static std::map<int, Agents*> agentsMap;
	static Dispatcher *dispatcher;/**< The object that handles communication with the GPU(s). */
};

} /* namespace mass */
