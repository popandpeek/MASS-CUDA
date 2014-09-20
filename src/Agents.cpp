/**
 *  @file Agents.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Agents.h"
#include "Dispatcher.h"

namespace mass {

Agents::~Agents() {
	if (NULL != agents) {
		delete[] this->agents;
	}
}

int Agents::getHandle() {
	return handle;
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
	return dispatcher->callAllAgents(handle, functionId, arguments, argSize, retSize);
}

void Agents::manageAll() {
	dispatcher->manageAllAgents(handle);
}

Agents::Agents(int handle, void *argument, int argument_size, Places *places,
		int initPopulation) {
	this->places = places;
	this->agents = NULL;
	this->handle = handle;
	this->numAgents = initPopulation;
	this->newChildren = 0;
	this->sequenceNum = 0;
}

//Places *places; /**< The places used in this simulation. */
//
//	Agent* agents; /**< The agents elements.*/
//
//	int handle; /**< Identifies the type of agent this is.*/
//
//	int numAgents; /**< Running count of living agents in system.*/
//
//	int newChildren; /**< Added to numAgents and reset to 0 each manageAll*/
//
//	int sequenceNum; /*!< The number of agents created overall. Used for agentId creation. */

}// mass namespace
