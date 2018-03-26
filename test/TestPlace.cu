#include <stdio.h> //remove after debugging
#include "TestPlace.h"

using namespace std;
using namespace mass;

MASS_FUNCTION TestPlace::TestPlace(PlaceState* state, void *argument) :
		Place(state, argument) {
	myState = (TestPlaceState*) state;
}


MASS_FUNCTION void TestPlace::findSpawningDest() { 

    int idx = myState -> index;
    int size = myState -> size[0];

    if (idx - size >= 0) { //top is within bounds
    	TestPlace* top = ((TestPlace*)myState->neighbors[0]);
        if (top->getAgentPopulation() == 0) {
            myState->spawningDest = top;
        }
    } 
}



MASS_FUNCTION TestPlace::~TestPlace() {
	// nothing to delete
}

MASS_FUNCTION void TestPlace::callMethod(int functionId, void *argument) {
	switch (functionId) {
		case FIND_SPAWNING_DEST:
			findSpawningDest();
			break;
		default:
			break;
	}
}
