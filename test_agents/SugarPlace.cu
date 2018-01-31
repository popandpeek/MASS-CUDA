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

MASS_FUNCTION void SugarPlace::incSugarAndPollution() {
    if ( myState->curSugar < myState->maxSugar )
    {
        myState->curSugar++;
    }
    myState->pollution += 1.0;
}

// Calculates average pollution between 4 neighbors
MASS_FUNCTION void SugarPlace::avePollutions() { 
    
    int idx = myState -> index;
    int size = myState -> size[0];

    double top, right, bottom, left;

    if (idx - size >= 0) { //top
    	if (myState->inMessages[0] == NULL) {
    		printf("top neighbor is NULL !!!!!!!!!!!!!!! idx = %d\n", idx);
    	}
    	top = *((double*) myState->inMessages[0]);
    } else {
    	top = 0.0;
    }
	
	if ((idx +1) % size != 0) { //right
    	if (myState->inMessages[1] == NULL) {
    		printf("right neighbor is NULL !!!!!!!!!!!!!!!idx = %d\n", idx);
    	}
    	right = *((double*) myState->inMessages[1]);
    } else {
    	right = 0.0;
    }

    if (idx + size < size*size) { //bottom
    	if (myState->inMessages[2] == NULL) {
    		printf("bottom neighbor is NULL !!!!!!!!!!!!!!!idx = %d\n", idx);
    	}
    	bottom = *((double*) myState->inMessages[2]);
    } else {
    	bottom = 0.0;
    }

    if (idx % size != 0) { //left
    	if (myState->inMessages[3] == NULL) {
    		printf("left neighbor is NULL !!!!!!!!!!!!!!!idx = %d\n", idx);
    	}
    	left = *((double*) myState->inMessages[3]);
    } else {
    	left = 0.0;
    }

    // idx + size < size*size ? top = *((double*) myState->inMessagesinMessages[0]) : 0.0;
    // idx - size >= 0 ? bottom = *((double*) myState->inMessages[2]) : 0.0;
    // (idx +1) % size != 0 ? right = *((double*) myState->inMessages[1]) : 0.0;
    // (idx - 1) % size != size-2 ? left = *((double*) myState->inMessages[3]) : 0.0;

    // printf("idx = [%d], top = %f, bottom = %f, left = %f, right = %f\n", idx, top, bottom, left, right);
    // printf("average pollution for idx[%d]: %f\n", idx, ( top + bottom + left + right ) / 4.0);

    myState->avePollution = ( top + bottom + left + right ) / 4.0;
}

MASS_FUNCTION void SugarPlace::updatePollutionWithAverage() {
	//printf("Inside updatePollutionWithAverage() for idx = %d. old pollution = %f, new pollution = %f\n", myState -> index, myState->pollution, myState->avePollution);
    myState->pollution = myState->avePollution;
    myState->avePollution = 0.0;
}


MASS_FUNCTION SugarPlace::~SugarPlace() {
	// nothing to delete
}

/**
 *  Gets a pointer to this place's out message.
 */
MASS_FUNCTION void *SugarPlace::getMessage() {
	return &(myState->pollution);
}

MASS_FUNCTION int SugarPlace::getCurSugar() {
	return myState->curSugar;
}

MASS_FUNCTION void SugarPlace::setCurSugar(int newSugar) {
    myState->curSugar = newSugar;
}

MASS_FUNCTION double SugarPlace::getPollution() {
    return myState -> pollution;
}

MASS_FUNCTION void SugarPlace::setPollution(double newPollution){
    myState -> pollution = newPollution;
}

MASS_FUNCTION void SugarPlace::callMethod(int functionId, void *argument) {
	switch (functionId) {
		case SET_SUGAR:
			setSugar();
			break;
		case INC_SUGAR_AND_POLLUTION:
			incSugarAndPollution();
			break;
		case AVE_POLLUTIONS:
			avePollutions();
			break;
		case UPDATE_POLLUTION_WITH_AVERAGE:
			updatePollutionWithAverage();
			break;
		default:
			break;
	}
}
