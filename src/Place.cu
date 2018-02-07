
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

	for (int i=0; i< N_DESTINATIONS; i++) {
		state->potentialNextAgents[i] = NULL;
		state->potentialNextAgentsIdxs[i] = -1;
	}
}


MASS_FUNCTION PlaceState* Place::getState() {
	return state;
}

MASS_FUNCTION void Place::resolveMigrationConflicts() {
	// printf("resolveMigrationConflicts() kernel for idx =%d\n", getIndex());

	if (MAX_AGENTS == 1) { //common case, easier computation

		Agent* acceptedAgent = NULL;
		int acceptedIdx = -1;

		for (int i=0; i< N_DESTINATIONS; i++) {
			// TODO: take out the index array and see if it detriments the performance at all
			// printf("resolveMigrationConflicts() kernel for idx =%d, inside the loop i=%d\n", getIndex(), i);

			if (state->potentialNextAgentsIdxs[i] != -1) {
				// printf("resolveMigrationConflicts() kernel for idx =%d, inside the loop i=%d. past if. potentialNextAgentsIdxs[i] = %d\n", getIndex(), i, state->potentialNextAgentsIdxs[i]);
				if ((acceptedIdx == -1) || (state->potentialNextAgentsIdxs[i] < acceptedIdx)){
					// printf("resolveMigrationConflicts() kernel for idx =%d, inside the loop i=%d. past second if\n", getIndex(), i);
					acceptedAgent = state->potentialNextAgents[i];
					acceptedIdx = state->potentialNextAgentsIdxs[i];
				}
			} 
		}


		if (acceptedAgent != NULL) {
			state->agents[0] = acceptedAgent;
			state->agentPop ++;
			// printf("place %d accepted agent %d\n", getIndex(), acceptedAgent -> getIndex());
		}
	} 

	else { // more than 1 agent can reside in a place
		printf("MAX_AGENTS > 1 in resolveMigrationConflicts()!!!\n");
		Agent* potentialResidents[N_DESTINATIONS];
		int potentialResidentsIdxs[N_DESTINATIONS] = {-1};

		for (int i=0; i< N_DESTINATIONS; i++) {
			if (state->potentialNextAgentsIdxs[i] == -1) continue;
			
			//Insert agent into proper place:
			for (int j=0; j< N_DESTINATIONS; j++) {
				if (potentialResidentsIdxs[j] == -1) {
					potentialResidentsIdxs[j] = state->potentialNextAgentsIdxs[i];
					potentialResidents[j] = state->potentialNextAgents[i];
					break;
				}

				if (state->potentialNextAgentsIdxs[i] < potentialResidentsIdxs[j]) {
					// insert new index here and shift everything right:
					for (int k=N_DESTINATIONS-1; k>j; k--) {
						potentialResidents[k] = potentialResidents[k-1];
						potentialResidentsIdxs[k] = potentialResidentsIdxs[k-1];
					}
					potentialResidents[j] = state->potentialNextAgents[i];
					potentialResidentsIdxs[j] = state->potentialNextAgentsIdxs[i];
					break;
				}
			}
		}

		// Copy the N_DESTINATIONS first agents into next agents array:
		int curAgent =0;
		while ((state->agentPop < MAX_AGENTS) && (potentialResidentsIdxs[curAgent] != -1)) {
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

	// Clean potentialNextAgents and potentialNextAgentsIdxs arrays
	for (int i=0; i< N_DESTINATIONS; i++) {
		state->potentialNextAgents[i] = NULL;
		state->potentialNextAgentsIdxs[i] = -1;
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
	state -> potentialNextAgentsIdxs[relativeIdx] = agent -> getIndex();
}


} /* namespace mass */

