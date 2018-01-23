/*
 *  @file Metal.cpp
 	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <ctime> // clock_t,
#include <stdio.h> //remove after debugging
#include "SugarPlace.h"

static const int maxMtSugar = 4; //max level of sugar in mountain peak

using namespace std;
using namespace mass;

MASS_FUNCTION SugarPlace::SugarPlace(PlaceState* state, void *argument) :
		Place(state, argument) {
	myState = (SugarPlaceState*) state;

    myState -> pollution = 0.0;    // the current pollution
    myState -> avePollution = 0.0; // averaging four neighbors' pollution
    myState -> nextAgentIdx = -1;      // the next agent to come here

    myState -> curSugar = 0;
    myState -> maxSugar = 0;
    myState -> nAgentsInPlace = 0;

    // Agent properties:
    // destinationIdx[idx] = -1; // the next place to migrate to
}

MASS_FUNCTION void SugarPlace::setSugar() {
	int mtCoord[2];
    int size = myState->size[0];

    mtCoord[0] = size/3;
    mtCoord[1] = size - size/3 - 1;
    
    int mt1 = initSugarAmount(myState ->index, size, mtCoord[0], mtCoord[1], maxMtSugar);
    int mt2 = initSugarAmount(myState ->index, size, mtCoord[1], mtCoord[0], maxMtSugar);
    
    myState -> curSugar = mt1 > mt2 ? mt1 : mt2;
    myState -> maxSugar = mt1 > mt2 ? mt1 : mt2;
}

MASS_FUNCTION int SugarPlace::initSugarAmount(int idx, int size, int mtPeakX, int mtPeakY, int maxMtSug) {
    int x_coord = idx % size;
    int y_coord = idx / size;  //division by 0
    
    float distance = sqrt((float)(( mtPeakX - x_coord ) * ( mtPeakX - x_coord ) + (mtPeakY - y_coord) * (mtPeakY - y_coord)));

    // radius is assumed to be simSize/2.
    int r = size/2;
    if ( distance < r )
    {
        // '+ 0.5' for rounding a value.
        return ( int )( maxMtSug + 0.5 - maxMtSug / ( float )r * distance );
    }
    else
        return 0;
}

MASS_FUNCTION SugarPlace::~SugarPlace() {
	// nothing to delete
}

/**
 *  Gets a pointer to this place's out message.
 */
 MASS_FUNCTION void *SugarPlace::getMessage() {
	return &(myState->curSugar);
}

MASS_FUNCTION void SugarPlace::callMethod(int functionId, void *argument) {
	switch (functionId) {
		case SET_SUGAR:
			setSugar();
			break;
	// case APPLY_HEAT:
	// 	applyHeat();
	// 	break;
	// case GET_VALS:
	// 	getVals();
	// case SET_BORDERS:
	// 	setBorders((myState->p));
	// case EULER_METHOD:
	// 	eulerMethod();
	// 	break;
	// case NEXT_PHASE:
	// 	nextPhase();
	// 	break;
		default:
			break;
	}
}
