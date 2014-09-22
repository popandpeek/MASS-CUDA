/**
 *  @file Agents_Base.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Agents_Base.h"
#include "Dispatcher.h"

namespace mass {

Agents_Base::~Agents_Base() {
}

int Agents_Base::getHandle() {
	return handle;
}

int Agents_Base::nAgents() {
	return numAgents;
}

Agents_Base::Agents_Base(int handle, void *argument, int argument_size,
		Places_Base *places, int initPopulation) {
	this->places = places;
	this->handle = handle;
	this->numAgents = initPopulation;
	this->newChildren = 0;
	this->sequenceNum = 0;
}

} // mass namespace
