/**
 *  @file Mass.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Dispatcher.h"
#include "Mass.h"

namespace mass {

// static initialization
Dispatcher *Mass::dispatcher = new Dispatcher();

void Mass::init(std::string args[], int ngpu) {
	Mass::dispatcher->init(ngpu);
}

void Mass::init(std::string args[]) {
	Mass::dispatcher = new Dispatcher();
	// 0 is the flag to use all available GPU resources
	Mass::dispatcher->init(0);
}

void Mass::finish() {
	delete Mass::dispatcher;
}

Places_Base *Mass::getPlaces(int handle) {
	return agentsMap.find(handle)->second;
}

Agents_Base *Mass::getAgents(int handle) {
	return placesMap.find(handle)->second;
}

} /* namespace mass */
