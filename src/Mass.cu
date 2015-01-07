/**
 *  @file Mass.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <time.h>
#include "Mass.h"
//#include "Agents.h"
//#include "Places.h"
//#include "Dispatcher.h"
#include "Logger.h"

using namespace std;

namespace mass {

// static initialization
Dispatcher *Mass::dispatcher = new Dispatcher();
map<int, Places*> Mass::placesMap;
map<int, Agents*> Mass::agentsMap;

void Mass::init(string args[], int &ngpu) {
	Logger::debug("Initializing Mass");
	if (NULL == Mass::dispatcher) {
		Mass::dispatcher = new Dispatcher();
	}
	Mass::dispatcher->init(ngpu);
}

void Mass::init(string args[]) {
	Logger::debug("Initializing Mass");
	if (NULL == Mass::dispatcher) {
		Mass::dispatcher = new Dispatcher();
	}
	// 0 is the flag to use all available GPU resources
	int flag = 0;
	Mass::dispatcher->init(flag);
}

void Mass::finish() {
	Logger::debug("Finishing Mass");
	delete Mass::dispatcher;
	Mass::dispatcher = NULL;
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
