/*
 *  @file Heat2d.h
 *  @author Nate Hart
 *	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef HEAT2D_H_
#define HEAT2D_H_

#include "Metal.h"
#include "../src/Places.h"

class Heat2d {

public:
	Heat2d();
	virtual ~Heat2d();

	void runMain();
	void displayResults(mass::Places *places, int time, int *placesSize);

};

#endif /* HEAT2D_H_ */
