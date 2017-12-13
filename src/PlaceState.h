/**
 * PlaceState.h
 *
 *  Author: Nate Hart
 *  Created on: Nov 18, 2014
 */

#ifndef PLACESTATE_H_
#define PLACESTATE_H_

//#include "Agent.h"

#define MAX_AGENTS 4
#define MAX_NEIGHBORS 8
#define MAX_DIMS 6

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
};

} /* namespace mass */
#endif /* PLACESTATE_H_ */
