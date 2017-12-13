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

void Mass::init(string args[]) {
	Logger::debug("Initializing Mass");
	if (dispatcher == NULL) {
		dispatcher = new Dispatcher();
	}
	dispatcher->init();
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

} /* namespace mass */
