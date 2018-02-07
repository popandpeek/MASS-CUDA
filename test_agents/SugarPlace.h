
#ifndef SUGAR_PLACE_H_
#define SUGAR_PLACE_H_

#include "../src/Place.h"
#include "../src/Logger.h"
#include "SugarPlaceState.h"

const static int maxVisible = 3;
const static int nMigrationDestinations = 6;

class SugarPlace: public mass::Place {

public:

	const static int SET_SUGAR = 0;
	const static int INC_SUGAR_AND_POLLUTION = 1;
	const static int AVE_POLLUTIONS = 2;
	const static int UPDATE_POLLUTION_WITH_AVERAGE = 3;
	const static int FIND_MIGRATION_DESTINATION = 4;
	const static int SELECT_AGENT_TO_ACCEPT = 5;
	const static int IDENTIFY_IF_GOOD_FOR_MIGRATION = 7;

	MASS_FUNCTION SugarPlace(mass::PlaceState *state, void *argument);
	MASS_FUNCTION ~SugarPlace();

	MASS_FUNCTION virtual void *getMessage();
	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);

	MASS_FUNCTION int getCurSugar();
	MASS_FUNCTION void setCurSugar(int newSugar);
	MASS_FUNCTION double getPollution();
	MASS_FUNCTION void setPollution(double newPollution);
	MASS_FUNCTION bool isGoodForMigration();
	MASS_FUNCTION SugarPlace* getMigrationDest();
	MASS_FUNCTION int getMigrationDestRelIdx();
	

private:

	SugarPlaceState* myState;

	MASS_FUNCTION int initSugarAmount(int idx, int size, int mtPeakX, int mtPeakY, int maxMtSug);
	MASS_FUNCTION void setSugar();
	MASS_FUNCTION void incSugarAndPollution();
	MASS_FUNCTION void avePollutions();
	MASS_FUNCTION void updatePollutionWithAverage();
	MASS_FUNCTION void findMigrationDestination();
	MASS_FUNCTION void identifyIfGoodForMigration();
};

#endif /* SUGAR_PLACE_H_ */
