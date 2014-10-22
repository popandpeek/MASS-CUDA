/**
 *  @file Mass.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <time.h>
#include "Mass.h"
#include "Agents.h"
#include "Places.h"
#include "Dispatcher.h"

using namespace std;

namespace mass {

// static initialization
Dispatcher *Mass::dispatcher = new Dispatcher();
map<int, Places*> Mass::placesMap;
map<int, Agents*> Mass::agentsMap;
Logger Mass::logger;

void Mass::init(string args[], int ngpu) {
	Mass::logger.debug("Initializing Mass");
	Mass::dispatcher->init(ngpu);
}

void Mass::init(string args[]) {
	Mass::logger.debug("Initializing Mass");
	Mass::dispatcher = new Dispatcher();
	// 0 is the flag to use all available GPU resources
	Mass::dispatcher->init(0);
}

void Mass::finish() {
	Mass::logger.debug("Finishing Mass");
	delete Mass::dispatcher;
}

Places *Mass::getPlaces(int handle) {
	return Mass::placesMap.find(handle)->second;
}

int Mass::numPlacesInstances() {
	return Mass::placesMap.size();
}

Agents *Mass::getAgents(int handle) {
	return Mass::agentsMap.find(handle)->second;
}

int Mass::numAgentsInstances() {
	return Mass::agentsMap.size();
}

} /* namespace mass */
