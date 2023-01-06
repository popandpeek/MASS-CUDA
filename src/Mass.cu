
#include <time.h>
#include "Mass.h"
#include "Logger.h"

using namespace std;

namespace mass {

// static initialization
Dispatcher *Mass::dispatcher = new Dispatcher();
map<int, Places*> Mass::placesMap;
map<int, Agents*> Mass::agentsMap;

void Mass::init() {
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

int* Mass::getRandomNumbers(int size, int max_num) {
	return dispatcher->calculateRandomNumbers(size, max_num);
}

} /* namespace mass */
