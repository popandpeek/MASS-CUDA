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
    	top = *((double*) myState->inMessages[0]);
    } else {
    	top = 0.0;
    }
	
	if ((idx +1) % size != 0) { //right
    	right = *((double*) myState->inMessages[1]);
    } else {
    	right = 0.0;
    }

    if (idx + size < size*size) { //bottom
    	bottom = *((double*) myState->inMessages[2]);
    } else {
    	bottom = 0.0;
    }

    if (idx % size != 0) { //left
    	left = *((double*) myState->inMessages[3]);
    } else {
    	left = 0.0;
    }

    myState->avePollution = ( top + bottom + left + right ) / 4.0;
}

MASS_FUNCTION void SugarPlace::updatePollutionWithAverage() {
    myState->pollution = myState->avePollution;
    myState->avePollution = 0.0;
}

MASS_FUNCTION void SugarPlace::findMigrationDestination() {
    // TODO: experiment if moving calulation of suitability into separate function will help 
    // TODO: experiment with moving this check to Ant class, thus will have less threads and less thread divergence
        
    if (getAgentPopulation() == 0) return;

    myState -> migrationDest = NULL; //initially assume we won't find a suitable place
    myState ->  migrationDestRelativeIdx = -1;

    int idx = myState -> index;
    int size = myState -> size[0];

    for(int i=0; i< maxVisible; i++) { //displacement to the right
        // TODO: fix the possibility to migrate across the plane border to next line
        if (idx + i + 1< size*size) {
            if (((SugarPlace *)(myState-> neighbors[i])) -> isGoodForMigration()) {
                myState -> migrationDest = (SugarPlace*)myState-> neighbors[i];
                myState -> migrationDestRelativeIdx = i;
                return;
            }
        }        
    }

    for (int i=maxVisible; i<maxVisible*2; i++) { //displacement up
        if (idx - (i-maxVisible+1)*size > 0) {
            if (((SugarPlace *)(myState-> neighbors[i])) -> isGoodForMigration()) {
                myState -> migrationDest = (SugarPlace*)myState-> neighbors[i];
                myState -> migrationDestRelativeIdx = i;
                return;
            }
        }
    }
}

MASS_FUNCTION void SugarPlace::identifyIfGoodForMigration() {
    if ((getAgentPopulation() == 0) && (myState->curSugar / ( 1.0 + myState->pollution) > 0.0)) {
        myState -> isGoodForMigration = true;
    } else {
        myState -> isGoodForMigration = false;
    }
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

MASS_FUNCTION bool SugarPlace::isGoodForMigration() {
    return myState -> isGoodForMigration;
}

MASS_FUNCTION SugarPlace* SugarPlace::getMigrationDest() {
    return myState -> migrationDest;
}

MASS_FUNCTION int SugarPlace::getMigrationDestRelIdx() {
    return myState -> migrationDestRelativeIdx;
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
        case FIND_MIGRATION_DESTINATION:
            findMigrationDestination();
            break;
        case IDENTIFY_IF_GOOD_FOR_MIGRATION:
            identifyIfGoodForMigration();
            break;
		default:
			break;
	}
}
