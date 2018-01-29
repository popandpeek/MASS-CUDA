
#ifndef PLACESTATE_H_
#define PLACESTATE_H_

#define MAX_AGENTS 4
#define MAX_NEIGHBORS 8
#define MAX_DIMS 6

class Agent; //forward declaration

namespace mass {

class PlaceState {
	friend class Place;

public:
	Place *neighbors[MAX_NEIGHBORS];  // my neighbors
	unsigned index;            // the row-major index of this place
	int size[MAX_DIMS];   // the size of the Places matrix
	char numDims;
	int message_size;  // the number of bytes in a message
	void *inMessages[MAX_NEIGHBORS]; // holds a pointer to each neighbor's outmessage.

    Agent *agents[MAX_AGENTS]; //agents residing on this place
    unsigned agentPop; // the population of agents on this place
};

} /* namespace mass */
#endif /* PLACESTATE_H_ */
