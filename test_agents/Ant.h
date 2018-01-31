
#ifndef ANT_H_
#define ANT_H_

#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "AntState.h"
#include "SugarPlace.h"

#define maxMetabolism 4;
#define maxInitAgentSugar 10;

class Ant: public mass::Agent {

public:

    const static int METABOLIZE = 0;
    const static int SET_INIT_VALUES = 1;

    MASS_FUNCTION Ant(mass::AgentState *state, void *argument);
    MASS_FUNCTION ~Ant();

    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual AntState* getState();

    // MASS_FUNCTION virtual SugarPlace* getPlace();
    // MASS_FUNCTION virtual void setPlace(SugarPlace* place);

private:

    AntState* myState;

    MASS_FUNCTION void metabolize();
    MASS_FUNCTION void setInitValues();
};

#endif /* ANT_H_ */
