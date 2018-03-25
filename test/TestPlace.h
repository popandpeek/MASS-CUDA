
#ifndef TEST_PLACE_H_
#define TEST_PLACE_H_

#include "../src/Place.h"
#include "../src/Logger.h"
#include "../src/PlaceState.h"

class TestPlace;

class TestPlaceState: public mass::PlaceState {
public:
    TestPlace* spawningDest;  //available spawning destination to the north from this place
};


class TestPlace: public mass::Place {

public:

	const static int FIND_SPAWNING_DEST = 0;

	MASS_FUNCTION TestPlace(mass::PlaceState *state, void *argument);
	MASS_FUNCTION ~TestPlace();

	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);

private:

	TestPlaceState* myState;

	MASS_FUNCTION void findSpawningDest();
};

#endif /* TEST_PLACE_H_ */
