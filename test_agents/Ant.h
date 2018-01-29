
#ifndef ANT_H_
#define ANT_H_

#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "AntState.h"

class Ant: public mass::Agent {

public:

    // const static int SET_SUGAR = 0;
    // const static int INC_SUGAR_AND_POLLUTION = 1;
    // const static int AVE_POLLUTIONS = 2;
    // const static int UPDATE_POLLUTION_WITH_AVERAGE = 3;

    MASS_FUNCTION Ant(mass::AgentState *state, void *argument);
    MASS_FUNCTION ~Ant();

    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual AntState* getState();

    // MASS_FUNCTION virtual SugarPlace* getPlace();
    // MASS_FUNCTION virtual void setPlace(SugarPlace* place);

private:

    AntState* myState;

    // MASS_FUNCTION int initSugarAmount(int idx, int size, int mtPeakX, int mtPeakY, int maxMtSug);
    // MASS_FUNCTION void setSugar();
    // MASS_FUNCTION void incSugarAndPollution();
    // MASS_FUNCTION void avePollutions();
    // MASS_FUNCTION void updatePollutionWithAverage();
};

#endif /* ANT_H_ */
