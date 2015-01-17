/**
 *  @file Place.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Place.h"
#include "Agent.h"
#include "PlaceState.h"

namespace mass {

/**
 *  A contiguous space of arguments is passed
 *  to the constructor.
 */MASS_FUNCTION Place::Place(PlaceState *state, void *args) {
	this->state = state;
	state->index = 0;
	state->agentPop = 0;
	state->message_size = 0;
	memset(state->neighbors, 0, MAX_NEIGHBORS);
	memset(state->inMessages, 0, MAX_NEIGHBORS);
	memset(state->agents, 0, MAX_AGENTS);
	memset(state->size, 0, MAX_DIMS);
}

/**
 * Registers an agent with this place.
 * @param agent the agent that is self-registering.
 */MASS_FUNCTION void Place::addAgent(Agent *agent) {
	// this works because of unique migration pattern that prevents collisions.
	unsigned idx = state->agentPop++;
	if (idx >= MAX_AGENTS) {
		--state->agentPop; // TODO silent failure is a shitty way to deal with this
	} else {
		agent->placePos = idx;
		state->agents[idx] = agent;
	}
}

/**
 * Unregisters an agent with this place.
 * @param agent the agent that is self-unregistering.
 */MASS_FUNCTION void Place::removeAgent(Agent *agent) {
	unsigned idx = agent->placePos;
	state->agents[idx] = NULL;
	--state->agentPop;
}

MASS_FUNCTION void *Place::getMessage() {
	return NULL;
}

MASS_FUNCTION void Place::setState(PlaceState *s) {
	state = s;
}

MASS_FUNCTION PlaceState* Place::getState() {
	return state;
}

MASS_FUNCTION int Place::getIndex() {
	return state->index;
}

MASS_FUNCTION void Place::setIndex(int index) {
	state->index = index;
}

MASS_FUNCTION void Place::setSize(int *dimensions, int nDims) {
	for (int i = 0; i < nDims; ++i) {
		state->size[i] = dimensions[i];
	}
}

} /* namespace mass */

