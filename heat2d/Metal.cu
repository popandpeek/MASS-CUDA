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
} // end constructor

MASS_FUNCTION Metal::~Metal() {
	// nothing to delete
}

MASS_FUNCTION double Metal::getTemp() {
	return myState->temp[myState->p];
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
		setBorders((myState->p));
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
} // end applyHeat

MASS_FUNCTION void *Metal::getVals() { //GET_VALS
	return &(myState->temp[myState->p]);
}

// MASS_FUNCTION void *Metal::exchange(void *arg) { // EXCHANGE
// 	return &(myState->temp[myState->p]);
// }

MASS_FUNCTION void Metal::eulerMethod() { // EULER_METHOD
	int p = myState->p;
	int p2 = (p + 1) % 2;
	if (!isBorderCell()) {
		double north = ((Metal*)(myState->neighbors[0])) -> getTemp();
		double east =  ((Metal*)(myState->neighbors[1])) -> getTemp();
		double south = ((Metal*)(myState->neighbors[2])) -> getTemp();
		double west =  ((Metal*)(myState->neighbors[3])) -> getTemp();

		double curTemp = myState->temp[p];
		myState->temp[p2] = curTemp + myState->r * (east - 2 * curTemp + west)
				+ myState->r * (south - 2 * curTemp + north);
	} else {
		setBorders (p2);
	}

	nextPhase();
} // end eulerMethod

MASS_FUNCTION void Metal::nextPhase() {
	myState->p = (myState->p + 1) % 2;
}

MASS_FUNCTION void Metal::setBorders(int phase) {
	int x = myState->index;
	int size = myState->size[0];
	int idx;

	if (x<size) { // top border
		idx = 2; // south neighbor value
	} 

	if (x >= size * size - size) {  // bottom border
		idx = 0; // north neighbor value

	}
	if (x % size == 0) { // left border
		idx = 1;  // east neighbor value
	}
 	else if (x % size == size - 1) {  // right border
		idx = 3; // west neighbor value
	}

	myState->temp[phase] = ((Metal*)(myState->neighbors[idx])) -> getTemp();
} // end setBorders

MASS_FUNCTION inline bool Metal::isBorderCell() {
	int x = myState->index;
	int size = myState->size[0];
	return (x < size || x > size * size - size || x % size == 0
			|| x % size == size - 1);
}
