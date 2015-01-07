/**
 *  @file Mass.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <iostream>
#include <fstream> // ofstream
#include <stddef.h>
#include <map>

#include "Dispatcher.h"
#include "Agents.h"
#include "Places.h"

namespace mass {

class Mass {
    friend class Agents;
    friend class Places;

public:

	/**
	 *  Initializes the MASS environment. Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 *  @param ngpu the number of GPUs to use
	 */
	static void init(std::string args[], int &ngpu);

	/**
	 *  Initializes the MASS environment using all available GPU resources. 
	 *  Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 */
	static void init(std::string args[] = NULL);

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

	template <typename T>
	static Places* createPlaces(T instantiator, int handle, void *argument,
			int argSize,int dimensions, int size[], int boundary_width);

private:

    static std::map<int, Places*> placesMap;
    static std::map<int, Agents*> agentsMap;
	static Dispatcher *dispatcher; /**< The object that handles communication with the GPU(s). */
};

template <typename T>
Places* Mass::createPlaces(T instantiator, int handle, void *argument,
		int argSize,int dimensions, int size[], int boundary_width){

	Places *places = new Places(handle, "", argument, argSize,
			dimensions, size, boundary_width);
	places->setDispatcher(Mass::dispatcher);
	Mass::dispatcher->configurePlaces(places);
	Place** p = Mass::dispatcher->instantiatePlaces(instantiator,argument, argSize,
			handle, places->numElements);
	places->setDevicePlaces(p);
	placesMap[handle] = places;
	return places;
}

} /* namespace mass */
