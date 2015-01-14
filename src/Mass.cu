/**
 *  @file Mass.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <time.h>
#include "Mass.h"
#include "Logger.h"

using namespace std;

namespace mass {

// static initialization
Dispatcher *Mass::dispatcher = new Dispatcher();
map<int, Places*> Mass::placesMap;
map<int, Agents*> Mass::agentsMap;

void Mass::init(string args[], int &ngpu) {
	Logger::debug("Initializing Mass");
	if (NULL == dispatcher) {
		dispatcher = new Dispatcher();
	}
	dispatcher->init(ngpu);
}

void Mass::init(string args[]) {
	// 0 is the flag to use all available GPU resources
	int ngpu = 0;
	Mass::init(args, ngpu);
}

void Mass::finish() {
	Logger::debug("Finishing Mass");
	delete dispatcher;
	dispatcher = NULL;
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
