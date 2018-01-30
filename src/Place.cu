
#include "Place.h"	
#include "PlaceState.h"
#include <stdio.h>

namespace mass {

/**
 *  A contiguous space of arguments is passed
 *  to the constructor.
 */MASS_FUNCTION Place::Place(PlaceState *state, void *args) {
	this->state = state;
	this->state->index = 0;
	this->state->message_size = 0;
	memset(this->state->neighbors, 0, MAX_NEIGHBORS);
	memset(this->state->inMessages, 0, MAX_NEIGHBORS);
	memset(this->state->size, 0, MAX_DIMS);
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
		int dim = dimensions[i];
		state->size[i] = dim;
	}
}

MASS_FUNCTION bool Place::addAgent(Agent* agent) {
	// TODO: MAX_AGENTS should be set by user and should regulate the 
	// collision-free migration in case max number of agents is reached
	
	if (state->agentPop < MAX_AGENTS) {
		state->agents[state->agentPop] = agent;
		state->agentPop ++;
		return true;
	}
	return false;
}

MASS_FUNCTION void Place::removeAgent(Agent* agent) {
	for (int i=0; i< state->agentPop; i++) {
		if (state->agents[i] == agent) {
			//shift all agents left:
			for (int j=i; j<state->agentPop-1; j++) {
				state->agents[j] = state->agents[j+1];
			}
			state->agents[state->agentPop-1] = NULL;

			state->agentPop --;
			return;
		}
	}
}

MASS_FUNCTION int Place::getAgentPopulation() {
	return state->agentPop;
}


} /* namespace mass */

