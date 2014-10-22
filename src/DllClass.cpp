/**
 *  @file DllClass.cpp
 *  @author Prof. Fukuda, modified by Nate Hart
 *
 *  This file was adapted from the file DllClass.cpp in the Mass c++ library.
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <sstream>
#include <cuda_runtime.h>
#include "DllClass.h"
#include "Mass.h"
#include "Agent.h"
#include "Place.h"

using namespace std;

namespace mass {

DllClass::DllClass(string className):placeElements(NULL) {
	// For logging
//	stringstream ss;

	// Create "./className"
	int char_len = 2 + className.size() + 1;
	char dot_className[char_len];
	bzero(dot_className, char_len);
	strncpy(dot_className, "./", 2);
	strncat(dot_className, className.c_str(), className.size());

	// load a given class
	if ((dllHandle = dlopen(dot_className, RTLD_LAZY)) == NULL) {
//		ss << "class: " << dot_className << " not found. Exiting program.";
		Mass::logger.debug("class: %s not found. Exiting program.", dot_className);
		exit(-1);
	} else {
//		ss << "class: " << dot_className << " was located. Program execution should continue.";
		Mass::logger.debug("class: %s was located. Program execution should continue.", dot_className);
	}

	// register the object instantiation/destroy functions
	instantiate = (instantiate_t *) dlsym(dllHandle, "instantiate");
	destroy = (destroy_t *) dlsym(dllHandle, "destroy");
	Mass::logger.debug("Done with DllClass constructor");
}

DllClass::~DllClass() {
	if (NULL != placeElements) {
		 free(placeElements);
	}

	for (int i = 0; i < agentElements.size(); ++i) {
		if (NULL != agentElements[i]) {
			free(agentElements[i]);
		}
	}
	agentElements.empty();
}

} /* namespace mass */
