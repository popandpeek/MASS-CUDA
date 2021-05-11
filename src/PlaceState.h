
#ifndef PLACESTATE_H
#define PLACESTATE_H

#include "settings.h"

class Agent; //forward declaration

namespace mass {

class PlaceState {
	friend class Place;

public:
	Place *neighbors[MAX_NEIGHBORS];  // my neighbors
	unsigned index;            // the row-major index of this place over the entire space 
	unsigned devIndex;		   // the row-major index of this place on a device
	int size[MAX_DIMS];   // the size of the Places matrix
	int devSize[MAX_DIMS]; // the size of the chunk of the Places matrix of the device that this Place resides
    Agent *agents[MAX_AGENTS]; //agents residing on this place
    unsigned agentPop; // the population of agents on this place

    Agent* potentialNextAgents[N_DESTINATIONS]; //agents that expressed an intention to migrate into this place
};

} /* namespace mass */
#endif /* PLACESTATE_H_ */