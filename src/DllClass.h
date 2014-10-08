/**
 *  @file DllClass.h
 *  @author Prof. Fukuda
 *
 *  This file was adapted from the file DllClass.h in the Mass c++ library.
 *  Documentation is by Nate Hart.
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#pragma once

#include <string>
#include <vector>
#include <dlfcn.h> // dlopen dlsym dlclose
#include "Agent.h"
#include "Place.h"

namespace mass {

class DllClass {
public:

	/**
	 * Storage for the return value from dlopen(), Used to create instantiate_t
	 * and destroy_t pointers.
	 */
	void *dllHandle;

	/**
	 * A pointer to the instantiate_t function of the class referenced by dllHandle.
	 */
	instantiate_t *instantiate;

	/**
	 * A pointer to the destroy_t function of the class referenced by dllHandle.
	 */
	destroy_t *destroy;

	/**
	 * Constructor for this class. Requires the name of a valid dll class.
	 * @param className the name of a valid user implemented class.
	 */
	DllClass(std::string className);

	// place instances
	void *placeElements;

	// agent instances
	std::vector<void*> agentElements;
};

} /* namespace mass */
