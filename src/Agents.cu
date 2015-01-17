/**
 *  @file Agents.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Agents.h"
#include "Agent.h"
#include "Mass.h"
#include "Places.h"
#include "Dispatcher.h"

namespace mass {

Agents::~Agents() {
	if (NULL != agentPtrs) {
		delete[] agentPtrs;
	}
}

int Agents::getHandle() {
	return handle;
}

int Agents::getPlacesHandle() {
	return places->getHandle();
}

Agent** Agents::getAgentElements() {
	agentPtrs = dispatcher->refreshAgents(handle);
	return agentPtrs;
}

int Agents::nAgents() {
	return numAgents;
}

void Agents::callAll(int functionId) {
	callAll(functionId, NULL, 0);
}

void Agents::callAll(int functionId, void *argument, int argSize) {
	dispatcher->callAllAgents(handle, functionId, argument, argSize);
}

void *Agents::callAll(int functionId, void *arguments[], int argSize,
		int retSize) {
	return dispatcher->callAllAgents(handle, functionId, arguments, argSize,
			retSize);
}

void Agents::manageAll() {
	dispatcher->manageAllAgents(handle);
}

Agents::Agents(int handle, void *argument, int argument_size, Places *places,
		int initPopulation) {
	this->places = places;
	this->handle = handle;
	this->numAgents = initPopulation;
	this->dispatcher = NULL;
	this->agentPtrs = NULL;
}

void Agents::setDispatcher(Dispatcher *d) {
	this->dispatcher = d;
}

} // mass namespace
