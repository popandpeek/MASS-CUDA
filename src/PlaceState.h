
#ifndef PLACESTATE_H_
#define PLACESTATE_H_

#define MAX_AGENTS 1
#define MAX_NEIGHBORS 8
#define MAX_DIMS 6
#define N_DESTINATIONS 6 //should match the nMigrationDestinations defined in the user app

class Agent; //forward declaration

namespace mass {

class PlaceState {
	friend class Place;

public:
	Place *neighbors[MAX_NEIGHBORS];  // my neighbors
	unsigned index;            // the row-major index of this place
	int size[MAX_DIMS];   // the size of the Places matrix

    Agent *agents[MAX_AGENTS]; //agents residing on this place
    unsigned agentPop; // the population of agents on this place

    Agent* potentialNextAgents[N_DESTINATIONS]; //agents that expressed an intention to migrate into this place

};

} /* namespace mass */
#endif /* PLACESTATE_H_ */
