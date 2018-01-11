/*
 *  @file Metal.h
 	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef METAL_H_
#define METAL_H_

#include "../src/Place.h"
#include "MetalState.h"

class Metal: public mass::Place {

public:

	const static int APPLY_HEAT = 0;
	const static int GET_VALS = 1;
	const static int EXCHANGE = 2;
	const static int EULER_METHOD = 3;
	const static int SET_BORDERS = 4;
	const static int NEXT_PHASE = 5;

	MASS_FUNCTION Metal(mass::PlaceState *state, void *argument);MASS_FUNCTION ~Metal();

	/**
	 *  Gets a pointer to this place's out message.
	 */
	MASS_FUNCTION virtual void *getMessage();

	/**
	 * Returns the number of bytes necessary to store this agent implementation.
	 * The most simple implementation is a single line of code:
	 * return sizeof(ThisClass);
	 *
	 * Because sizeof is respoved at compile-time, the user must implement this
	 * function rather than inheriting it.
	 *
	 * @return an int >= 0;
	 */
	//MASS_FUNCTION virtual int placeSize();

	// TODO remove this call if not necessary
	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);

	MASS_FUNCTION void nextPhase();

private:

	MetalState* myState;

	MASS_FUNCTION void applyHeat();
	MASS_FUNCTION void *getVals();
	MASS_FUNCTION void *exchange(void *arg);
	MASS_FUNCTION void eulerMethod();

	MASS_FUNCTION void setBorders(int phase);
	MASS_FUNCTION inline bool isBorderCell();
};

#endif /* METAL_H_ */
