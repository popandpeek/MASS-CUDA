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
	static void init( std::string args[], int ngpu );

	/**
	 *  Initializes the MASS environment using all available GPU resources. 
	 *  Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 */
	static void init( std::string args[] );

	/**
	 *  Shuts down the MASS environment, releasing all resources.
	 */
	static void finish();

	/**
	 *  Gets the places object for this handle.
	 *  @param handle an int that corresponds to a places object.
	 *  @return NULL if not found.
	 */
	static Places<Place> *getPlaces(int handle);

	/**
	 *  Gets the agents object for this handle.
	 *  @param handle an int that corresponds to an agents object.
	 *  @return NULL if not found.
	 */
	static Agents<Agent> *getAgents(int handle);


private:

	static Model *model; /**< The data model for this simulation. */
	static Dispatcher *dispatcher;/**< The object that handles communication with the GPU(s). */
};

} /* namespace mass */
#endif // MASS_H_
