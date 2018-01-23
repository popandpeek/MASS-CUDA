/*
 *  @file Heat2d.h
 	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef SUGARSCAPE_H_
#define SUGARSCAPE_H_

#include "SugarScape.h"
#include "../src/Places.h"

class SugarScape {

public:
	SugarScape();
	virtual ~SugarScape();

	void runMassSim(int size, int max_time, int interval);
	// void runDeviceSim(int size, int max_time, int interval);
	// void runHostSim(int size, int max_time, int interval);
	void displaySugar(mass::Places *places, int time, int *placesSize);

};

#endif /* SUGARSCAPE_H_ */
