#ifndef DENDRITE_H
#define DENDRITE_H


#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "GrowingEndState.h"
#include "NeuronPlace.h"


class GrowingEnd: public mass::Agent {

public:

    const static int INIT_AXNS = 0;
    const static int INIT_DENDRITES = 1;
    const static int SPAWN_AXONS = 2;
    const static int SPAWN_DENDRITES = 3;
    const static int MIGRATE = 4;

    MASS_FUNCTION GrowingEnd(mass::AgentState *state, void *argument = NULL);
    MASS_FUNCTION ~GrowingEnd();
    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual GrowingEndState* getState();

private:

    GrowingEndState* myState;

    MASS_FUNCTION void initAxons();
    MASS_FUNCTION void initDendrites();
    MASS_FUNCTION void spawnAxons();
    MASS_FUNCTION void spawnDendrites();

};

#endif