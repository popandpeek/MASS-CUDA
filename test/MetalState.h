/*
 *  @file MetalState.h
 	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef METALSTATE_H_
#define METALSTATE_H_

#include "../src/PlaceState.h"

class MetalState: public mass::PlaceState {
public:

	double temp[2];  // this place's temperature
	int p; // the index of temp that holds the most recently calculated temperature
	double r; // a coefficient used in Euler's method
};

#endif /* METALSTATE_H_ */
