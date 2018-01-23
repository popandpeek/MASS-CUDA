
#ifndef SUGAR_PLACE_H_
#define SUGAR_PLACE_H_

#include "../src/Place.h"
#include "../src/Logger.h"
#include "SugarPlaceState.h"

class SugarPlace: public mass::Place {

public:

	const static int SET_SUGAR = 0;
	// const static int GET_VALS = 1;
	// const static int EXCHANGE = 2;
	// const static int EULER_METHOD = 3;
	// const static int SET_BORDERS = 4;
	// const static int NEXT_PHASE = 5;

	MASS_FUNCTION SugarPlace(mass::PlaceState *state, void *argument);
	MASS_FUNCTION ~SugarPlace();

	MASS_FUNCTION virtual void *getMessage();

	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);

private:

	SugarPlaceState* myState;

	MASS_FUNCTION int initSugarAmount(int idx, int size, int mtPeakX, int mtPeakY, int maxMtSug);
	MASS_FUNCTION void setSugar();

	// MASS_FUNCTION void applyHeat();
	// MASS_FUNCTION void *getVals();
	// MASS_FUNCTION void *exchange(void *arg);
	// MASS_FUNCTION void eulerMethod();

	// MASS_FUNCTION void setBorders(int phase);
	// MASS_FUNCTION inline bool isBorderCell();
};

#endif /* SUGAR_PLACE_H_ */
