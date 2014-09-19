/**
 *  @file Mass.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef MASS_H_
#define MASS_H_

#include <iostream>
#include <stddef.h>
#include <map>
#include "Agents.h"
#include "Places.h"

// forward declarations
class Dispatcher;
class Model;

#define WARP_SIZE 32    // threads per warp
#define BLOCK_SIZE 512  // max threads per block
namespace mass {

class Mass {
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
	static Places *getPlaces(int handle);

	/**
	 *  Gets the agents object for this handle.
	 *  @param handle an int that corresponds to an agents object.
	 *  @return NULL if not found.
	 */
	static Agents *getAgents(int handle);

	/**
	 * Creates a Places object with the specified parameters.
	 * @param handle the unique number that will identify this Places object.
	 * @param argument an argument that will be passed to all Places upon creation
	 * @param argSize the size in bytes of argument
	 * @param dimensions the number of dimensions in the places matrix
	 * (i.e. 1, 2, 3, etc...)
	 * @param size the size of each dimension ordered {numRows, numCols,
	 * numDeep, ...}. This must be dimensions elements long.
	 * @return a pointer the the created places object
	 */
	template<typename T>
	static Places *createPlaces(int handle, void *argument, int argSize,
			int dimensions, int size[]) {
		return dispatcher->createPlaces<T>(handle, argument, argSize,
				dimensions, size);
	}

	/**
	 * Creates an Agents object with the specified parameters.
	 * @param handle the unique number that will identify this Agents object.
	 * @param argument an argument that will be passed to all Agents upon creation
	 * @param argSize the size in bytes of argument
	 * @param places the Places object upon which this agents collection will operate.
	 * @param initPopulation the starting number of agents to instantiate
	 * @return
	 */
	template<typename T>
	static Agents *createAgents(int handle, void *argument, int argSize,
			Places *places, int initPopulation) {
		return dispatcher->createAgents<T>(handle, argument, argSize, places,
				initPopulation);
	}

private:

	static Model *model; /**< The data model for this simulation. */
	static Dispatcher *dispatcher;/**< The object that handles communication with the GPU(s). */
};

} /* namespace mass */
#endif // MASS_H_
