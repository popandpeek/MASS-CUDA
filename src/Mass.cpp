/**
 *  @file Mass.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Dispatcher.h"
#include "Mass.h"
#include "Model.h"

namespace mass {
// Model model; /**< The data model for this simulation. */
// Dispatcher dispatcher;/**< The object that handles communication with the GPU(s). */

void Mass::init(std::string args[], int ngpu) {
	Mass::model = new Model();
	Mass::dispatcher = new Dispatcher();
	Mass::dispatcher->init(ngpu, Mass::model);
}

void Mass::init(std::string args[]) {
	Mass::model = new Model();
	Mass::dispatcher = new Dispatcher();
	// 0 is the flag to use all available GPU resources
	Mass::dispatcher->init(0, Mass::model);
}

void Mass::finish() {
	delete Mass::model;
	delete Mass::dispatcher;
}

Places<Place> *Mass::getPlaces(int handle) {
	return Mass::model->getPlaces(handle);
}

Agents<Agent> *Mass::getAgents(int handle) {
	return Mass::model->getAgents(handle);
}

} /* namespace mass */
