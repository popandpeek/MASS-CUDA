#ifndef DENDRITE_H
#define DENDRITE_H


#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "GrowingEndState.h"
#include "NeuronPlace.h"


class GrowingEnd: public mass::Agent {

public:

    const static int INIT_AXNOS = 0;
    const static int INIT_DENDRITES = 1;
    const static int SET_SPAWN_TIME = 2;
    const static int SPAWN_AXONS = 3;
    const static int SPAWN_DENDRITES = 4;
    const static int GROW_AXON_SOMA = 5;
    const static int GROW_DENDRITE_SOMA = 6;
    const static int AXON_TO_SYNAPSE = 7;
    const static int GROW_AXONS_OUTSIDE_SOMA = 8;
    const static int BRANCH_SYNAPSES = 9;
    const static int BRANCH_DENDRITES = 10;
    const static int GROW_SYNAPSE = 11;
    const static int GROW_DENDRITE = 12;
    const static int SET_BRANCHED_DENDRITES = 13;
    const static int SET_MIGRATED_BRANCHES = 14;
    const static int GET_SYNAPSE_SIGNAL = 15;
    const static int CONNECTION_TRAVEL = 16;
    const static int UPDATE_ITER = 17;


    MASS_FUNCTION GrowingEnd(mass::AgentState *state, void *argument = NULL);
    MASS_FUNCTION ~GrowingEnd();
    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual GrowingEndState* getState();

private:

    GrowingEndState* myState;

    MASS_FUNCTION void initAxons(int*);
    MASS_FUNCTION void initDendrites(int*);
    MASS_FUNCTION void setSpawnTime();
    MASS_FUNCTION void spawnAxon();
    MASS_FUNCTION void spawnDendrite();
    MASS_FUNCTION void growFromSoma();
    MASS_FUNCTION void axonToSynapse(int*);
    MASS_FUNCTION void growAxonsNotSoma(int*);
    MASS_FUNCTION void branchSynapses(int*);
    MASS_FUNCTION void branchDendrites(int*);
    MASS_FUNCTION void growSynapse();
    MASS_FUNCTION void growDendrite();
    MASS_FUNCTION void setBranchedSynapses();
    MASS_FUNCTION void setBranchedDendrites();
    MASS_FUNCTION void setMigratedBranches();
    MASS_FUNCTION void getSynapseSignal();
    MASS_FUNCTION void connectionTravel();
    MASS_FUNCTION void updateIters();

};

#endif