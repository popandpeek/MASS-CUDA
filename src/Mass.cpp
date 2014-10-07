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
ofstream Mass::logger;
time_t Mass::rawtime;
struct tm * Mass::ptm;

void Mass::init(string args[], int ngpu) {
	Mass::dispatcher->init(ngpu);
	Mass::log("Initializing Mass");
}

void Mass::init(string args[]) {
	Mass::dispatcher = new Dispatcher();
	// 0 is the flag to use all available GPU resources
	Mass::dispatcher->init(0);
	Mass::log("Initializing Mass");
}

void Mass::finish() {
	delete Mass::dispatcher;
	Mass::log("Finishing Mass");
	Mass::logger.close();
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

Places *Mass::createPlaces(int handle, std::string classname, void *argument,
		int argSize, int dimensions, int size[], int boundary_width) {
	Places *places = Mass::dispatcher->createPlaces(handle, classname, argument,
			argSize, dimensions, size, boundary_width);
	if (NULL != places) {
		placesMap[handle] = places;
	}
	return places;
}

Agents *Mass::createAgents(int handle, std::string classname, void *argument,
		int argSize, Places *places, int initPopulation) {
	Agents *agents = Mass::dispatcher->createAgents(handle, classname, argument,
			argSize, places, initPopulation);
	if (NULL != agents) {
		agentsMap[handle] = agents;
	}
	return agents;
}

void Mass::log(std::string message) {
	// get local time
	time(&rawtime);
	ptm = localtime(&rawtime);
	const size_t BUFSIZE = 80;
	char buf[BUFSIZE];

	if (!Mass::logger.is_open()) {
		setLogFile("mass_log.txt");
	}

	strftime(buf, BUFSIZE, "%H:%M:%S ", ptm); // record time
	Mass::logger << buf << message << "\n"; // log message
}

void Mass::setLogFile(std::string filename) {

	// get local time
	time(&rawtime);
	ptm = localtime(&rawtime);
	const size_t BUFSIZE = 80;
	char buf[BUFSIZE];

	if (Mass::logger.is_open()) {
		strftime(buf, BUFSIZE, "%H:%M:%S ", ptm); // record time
		Mass::logger << buf << "Log file switched to " << filename << endl;
		Mass::logger.close();
	}

	Mass::logger.open(filename.c_str(), ios::out | ios::app); // open log file
	strftime(buf, BUFSIZE, "%a %Y/%m/%d %H:%M:%S ", ptm); // record detailed time
	Mass::logger << "\n\n" << buf << " Logger initialized\n"; // log init

}

} /* namespace mass */
