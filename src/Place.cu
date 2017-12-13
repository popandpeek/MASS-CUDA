/**
 *  @file Place.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Place.h"	
#include "PlaceState.h"

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

} /* namespace mass */

