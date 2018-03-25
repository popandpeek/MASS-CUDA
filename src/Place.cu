
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
	memset(this->state->neighbors, 0, MAX_NEIGHBORS);
	memset(this->state->size, 0, MAX_DIMS);

	for (int i=0; i< N_DESTINATIONS; i++) {
		state->potentialNextAgents[i] = NULL;
	}
}


MASS_FUNCTION PlaceState* Place::getState() {
	return state;
}

MASS_FUNCTION void Place::resolveMigrationConflicts() {

	if (MAX_AGENTS == 1) { //common case, easier computation
		Agent* acceptedAgent = NULL;

		for (int i=0; i< N_DESTINATIONS; i++) {
			if (state->potentialNextAgents[i] != NULL) {
				if ((acceptedAgent == NULL) || (state->potentialNextAgents[i]->getIndex() < acceptedAgent->getIndex())){
					acceptedAgent = state->potentialNextAgents[i];
				}
			} 
		}


		if (acceptedAgent != NULL) {
			state->agents[0] = acceptedAgent;
			state->agentPop ++;
		}
	} 

	else { // more than 1 agent can reside in a place
		Agent* potentialResidents[N_DESTINATIONS];

		for (int i=0; i< N_DESTINATIONS; i++) {
			if (state->potentialNextAgents[i] == NULL) continue;
			
			//Insert agent into proper place:
			for (int j=0; j< N_DESTINATIONS; j++) {
				if (potentialResidents[j] == NULL) {
					potentialResidents[j] = state->potentialNextAgents[i];
					break;
				}

				if (state->potentialNextAgents[i]->getIndex() < potentialResidents[j]->getIndex()) {
					// insert new index here and shift everything right:
					for (int k=N_DESTINATIONS-1; k>j; k--) {
						potentialResidents[k] = potentialResidents[k-1];
					}
					potentialResidents[j] = state->potentialNextAgents[i];
					break;
				}
			}
		}

		// Copy the N_DESTINATIONS first agents into next agents array:
		int curAgent =0;
		while ((state->agentPop < MAX_AGENTS) && (potentialResidents[curAgent] != NULL)) {
			//copy agent into the first available spot:
			for (int i=0; i< MAX_AGENTS; i++) {
				if (state->agents[i] == NULL) {
					state->agents[i] = potentialResidents[curAgent];
					curAgent ++;
					state->agentPop ++;
					break;
				}
			}
		}
	}

	// Clean potentialNextAgents array
	for (int i=0; i< N_DESTINATIONS; i++) {
		state->potentialNextAgents[i] = NULL;
	}

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
	if (state->agentPop < MAX_AGENTS) {
		state->agents[state->agentPop] = agent;
		state->agentPop ++;
		return true;
	}
	return false;
}

MASS_FUNCTION void Place::removeAgent(Agent* agent) {
	// printf("   Inside Place::removeAgent for agent %d\n", getIndex());
	for (int i=0; i< state->agentPop; i++) {
		if (state->agents[i] == NULL) continue;
		if (state->agents[i]->getIndex() == agent->getIndex()) {
			//shift all agents left:
			for (int j=i; j < state->agentPop - 1; j++) {
				state->agents[j] = state->agents[j+1];
			}
			state->agents[state->agentPop-1] = NULL;

			// printf("decreasing agentPop for agent %d\n", getIndex());
			state->agentPop --;
			return;
		}
	}
	// TODO: for big sizes sometimes get this message, figure out why
	// printf("REQUESTED AGENT WASNT FOUND IN THE PLACE. agent id =%d, place id = %d\n", agent->getIndex(), getIndex());
}

MASS_FUNCTION int Place::getAgentPopulation() {
	return state->agentPop;
}


MASS_FUNCTION void Place::addMigratingAgent(Agent* agent, int relativeIdx) {
	// printf("Place::addMigratingAgent agent %d to place %d, relative idx = %d \n", agent->getIndex(), getIndex(), relativeIdx);
	state -> potentialNextAgents[relativeIdx] = agent;
}


} /* namespace mass */

