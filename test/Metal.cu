/*
 *  @file Metal.cpp
 *  @author Nate Hart
 *	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <ctime> // clock_t,
#include "Metal.h"

using namespace std;
using namespace mass;

MASS_FUNCTION Metal::Metal(PlaceState* state, void *argument) :
		Place(state, argument) {
	myState = (MetalState*) state;
	myState->r = *((double*) argument);

	// start out cold
	myState->temp[0] = 0.0;
	myState->temp[1] = 0.0;

	myState->p = 0;
} // end constructor

MASS_FUNCTION Metal::~Metal() {
	// nothing to delete
}

/**
 *  Gets a pointer to this place's out message.
 */MASS_FUNCTION void *Metal::getMessage() {
	return exchange(NULL);
}

/**
 * Returns the number of bytes necessary to store this agent implementation.
 * The most simple implementation is a single line of code:
 * return sizeof(ThisClass);
 *
 * Because sizeof is resolved at compile-time, the user must implement this
 * function rather than inheriting it.
 *
 * @return an int >= 0;
 */MASS_FUNCTION int Metal::placeSize() {
	return sizeof(Metal);
}

MASS_FUNCTION void Metal::callMethod(int functionId, void *argument) {
	switch (functionId) {
	case APPLY_HEAT:
		applyHeat();
		break;
	case GET_VALS:
		getVals();
	case EXCHANGE:
		exchange(argument);
	case EULER_METHOD:
		eulerMethod();
		break;
	default:
		break;
	}
} // end callMethod

MASS_FUNCTION void Metal::applyHeat() { // APPLY_HEAT
	int width = myState->size[0];
	// only heat first row
	if (myState->index < width) {
		int x = myState->index;

		if (x >= (width / 3) && x < (width / 3 * 2))
			myState->temp[myState->p] = 19.0;
	}
} // end applyHeat

MASS_FUNCTION void *Metal::getVals() { //GET_VALS
	return &(myState->temp[myState->p]);
}

MASS_FUNCTION void *Metal::exchange(void *arg) { // EXCHANGE
	return &(myState->temp[myState->p]);
} // end exchange

MASS_FUNCTION void Metal::eulerMethod() { // EULER_METHOD
	int p2 = (myState->p + 1) % 2;

	if (!isBorderCell()) {
		// perform forward Euler method
		double north = *((double*) myState->inMessages[0]);
		double east = *((double*) myState->inMessages[1]);
		double south = *((double*) myState->inMessages[2]);
		double west = *((double*) myState->inMessages[3]);

		double curTemp = myState->temp[myState->p];
		myState->temp[p2] = curTemp + myState->r * (east - 2 * curTemp + west)
				+ myState->r * (south - 2 * curTemp + north);

	} else {
		// copy neighbor's temp to p2 border
		setBorders(p2);
	}

	// phase always changes
	myState->p = p2;
} // end eulerMethod

MASS_FUNCTION void Metal::setBorders(int phase) {
	int x = myState->index;
	int size = myState->size[0];
	if (x < size)  // this is a top border
		myState->temp[phase] = *(double*) myState->inMessages[1]; // no north, so south will be 2nd element

	else if (x >= size * size - size)  // this is a bottom border
		myState->temp[phase] = *(double*) myState->inMessages[0]; // use north value

	else if (x % size == 0)  // this is a left border
		myState->temp[phase] = *(double*) myState->inMessages[1]; // use east value

	else if (x % size == size - 1)  // this is a right border
		myState->temp[phase] = *(double*) myState->inMessages[2]; // no east, so west is 3rd element

} // end setBorders

MASS_FUNCTION inline bool Metal::isBorderCell() {
	int x = myState->index;
	int size = myState->size[0];
	return (x < size || x > size * size - size || x % size == 0
			|| x % size == size - 1);
}
