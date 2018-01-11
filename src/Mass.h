
#pragma once

#include <iostream>
#include <fstream> // ofstream
#include <stddef.h>
#include <map>

#include "Dispatcher.h"
#include "Places.h"

namespace mass {

class Mass {
	friend class Places;

public:
	/**
	 *  Initializes the MASS environment using default GPU resource. 
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
	static Places *getPlaces(int handle);

	/**
	 *  Gets the number of Places collections in this simulation.
	 */
	static int numPlacesInstances();

	/**
	 * Creates a Places instance with the provided parameters.
	 */
	template<typename P, typename S>
	static Places* createPlaces(int handle, void *argument, int argSize,
			int dimensions, int size[]);


private:

	static std::map<int, Places*> placesMap;
	static Dispatcher *dispatcher; /**< The object that handles communication with the GPU(s). */
};

template<typename P, typename S>
Places* Mass::createPlaces(int handle, void *argument, int argSize,
		int dimensions, int size[]) {

	Logger::debug("Entering Mass::createPlaces\n");
	// create an API object for this Places collection
	Places *places = new Places(handle, dimensions, size, dispatcher);
	placesMap[handle] = places;

	// perform actual instantiation of user classes
	dispatcher->instantiatePlaces<P, S>(handle, argument, argSize, dimensions,
			size, places->numElements);
	Logger::debug("Exiting Mass::createPlaces\n");

	return places;
}


} /* namespace mass */
