
#ifndef ANT_H
#define ANT_H

#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "AntState.h"
#include "SugarPlace.h"

class Ant: public mass::Agent {

public:

    const static int METABOLIZE = 0;
    const static int SET_INIT_SUGAR = 1;
    const static int SET_INIT_METABOLISM = 2;
    const static int MIGRATE = 3;

    MASS_FUNCTION Ant(mass::AgentState *state, void *argument = NULL);
    MASS_FUNCTION ~Ant();
    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual AntState* getState();

private:

    AntState* myState;

    MASS_FUNCTION void metabolize();
    MASS_FUNCTION void setInitSugar(int *);
    MASS_FUNCTION void setInitMetabolism(int *);
    MASS_FUNCTION void migrate();
};

#endif /* ANT_H_ */
