/*
 *  @file Metal.cpp
 	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <ctime> // clock_t,
#include <stdio.h> //remove after debugging
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
	printf("Constructor for Metal. r=%d, p= %d\n", myState->r, myState->p);
} // end constructor

MASS_FUNCTION Metal::~Metal() {
	// nothing to delete
}

/**
 *  Gets a pointer to this place's out message.
 */MASS_FUNCTION void *Metal::getMessage() {
	return &(myState->temp[myState->p]);
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
 */
//  MASS_FUNCTION int Metal::placeSize() {
// 	return sizeof(Metal);
// }

MASS_FUNCTION void Metal::callMethod(int functionId, void *argument) {
	switch (functionId) {
	case APPLY_HEAT:
		applyHeat();
		break;
	case GET_VALS:
		getVals();
	// case EXCHANGE:
	// 	exchange(argument);
	case SET_BORDERS:
		setBorders((myState->p)); //+ 1) % 2);
	case EULER_METHOD:
		eulerMethod();
		break;
	case NEXT_PHASE:
		nextPhase();
		break;
	default:
		break;
	}
} // end callMethod

MASS_FUNCTION void Metal::applyHeat() { // APPLY_HEAT
	int width = myState->size[0];
	int x = myState->index;
	// only heat first row
	if (x < width) {

		if (x >= (width / 3) && x < (width / 3 * 2))
			myState->temp[myState->p] = 19.0;
	}
	printf("Running applyHeat kernel. width = %d, x = %d, temp after heating = %d\n", width, x, myState->temp[myState->p]);
} // end applyHeat

MASS_FUNCTION void *Metal::getVals() { //GET_VALS
	return &(myState->temp[myState->p]);
}

// MASS_FUNCTION void *Metal::exchange(void *arg) { // EXCHANGE
// 	return &(myState->temp[myState->p]);
// }

MASS_FUNCTION void Metal::eulerMethod() { // EULER_METHOD
	int p = myState->p;
	if (!isBorderCell()) {
		int p2 = (p + 1) % 2;

		double north = *((double*) myState->inMessages[0]);
		double east = *((double*) myState->inMessages[1]);
		double south = *((double*) myState->inMessages[2]);
		double west = *((double*) myState->inMessages[3]);

		double curTemp = myState->temp[p];
		myState->temp[p2] = curTemp + myState->r * (east - 2 * curTemp + west)
				+ myState->r * (south - 2 * curTemp + north);
		//printf("eulerMethod() kernel, thread is not a border cell, neighbors: north=%d, east=%d, south=%d, west=%d. Curent temp = %d, New temp = %d\n", north, east, south, west, curTemp, myState->temp[p2]);
	} else {
		//printf("eulerMethod() kernel, thread IS a border cell. Curent temp = %d\n", myState->temp[p]);
		setBorders (p);
	}

	nextPhase();
} // end eulerMethod

MASS_FUNCTION void Metal::nextPhase() {
	myState->p = (myState->p + 1) % 2;
}

MASS_FUNCTION void Metal::setBorders(int phase) {
	int x = myState->index;
	int size = myState->size[0];
	int idx = 1; // used for top and left borders

	if (x >= size * size - size) {  // this is a bottom border
		idx = 0; // use north value

	} else if (x % size == size - 1) {  // this is a right border
		idx = 2; // no east, so west is 3rd element
	}

	myState->temp[phase] = *((double*) myState->inMessages[idx]);
} // end setBorders

MASS_FUNCTION inline bool Metal::isBorderCell() {
	int x = myState->index;
	int size = myState->size[0];
	return (x < size || x > size * size - size || x % size == 0
			|| x % size == size - 1);
}
