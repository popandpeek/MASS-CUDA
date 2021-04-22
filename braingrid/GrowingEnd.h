#ifndef DENDRITE_H
#define DENDRITE_H


#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "GrowingEndState.h"
#include "NeuronPlace.h"
#include "BrainGridConstants.h"


class GrowingEnd: public mass::Agent {

public:

    const static int INIT_AXONS = 0;
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
    const static int SET_BRANCHED_SYNAPSES = 13;
    const static int SET_BRANCHED_DENDRITES = 14;
    const static int GROW_BRANCHES = 15;
    const static int SOMA_TRAVEL = 16;
    const static int GET_SIGNAL = 17;
    const static int DENDRITE_SOMA_TRAVEL = 18;
    const static int SET_SOMA_SIGNAL = 19;
    const static int UPDATE_ITERS = 20;

    MASS_FUNCTION GrowingEnd(mass::AgentState *state, void *argument = NULL);
    MASS_FUNCTION ~GrowingEnd();
    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual GrowingEndState* getState();
    MASS_FUNCTION bool isGrowing();
    MASS_FUNCTION void setGrowing(bool);
    MASS_FUNCTION int getType();
    MASS_FUNCTION int getSomaIndex();
    MASS_FUNCTION NeuronPlace* getSoma();
    MASS_FUNCTION void setSomaIndex(int);
    MASS_FUNCTION void setSoma(NeuronPlace*);
    MASS_FUNCTION int getConnectPlaceIndex();
    MASS_FUNCTION NeuronPlace* getConnectPlace();
    MASS_FUNCTION void setConnectPlaceIndex(int);
    MASS_FUNCTION void setConnectPlace(NeuronPlace*);
    MASS_FUNCTION double getSignal();
    MASS_FUNCTION void setSignal(double);
    MASS_FUNCTION bool justGrew();
    MASS_FUNCTION void setGrowthDirection(int);


private:

    GrowingEndState* myState;

    MASS_FUNCTION void initAxons();
    MASS_FUNCTION void initDendrites();
    MASS_FUNCTION void setDendriteSpawnTime();
    MASS_FUNCTION void setAxonSpawnTime();
    MASS_FUNCTION void spawnAxon();
    MASS_FUNCTION void spawnDendrite();
    MASS_FUNCTION void growFromSoma();
    MASS_FUNCTION void setNewlyGrownSynapse();
    MASS_FUNCTION void setNewlyGrownDendrites();
    MASS_FUNCTION void axonToSynapse(int*);
    MASS_FUNCTION void growAxonsNotSoma(int*);
    MASS_FUNCTION void branchSynapses(int*);
    MASS_FUNCTION void branchDendrites(int*);
    MASS_FUNCTION void growSynapsesNotSoma(int*);
    MASS_FUNCTION void growDendritesNotSoma(int*);
    MASS_FUNCTION void setBranchedSynapses();
    MASS_FUNCTION void setBranchedDendrites();
    MASS_FUNCTION void growBranches();
    MASS_FUNCTION void somaTravel();
    MASS_FUNCTION void neuronGetSignal();
    MASS_FUNCTION void dendriteSomaTravel();
    MASS_FUNCTION void setSomaSignal();
    MASS_FUNCTION void updateIters();
};

#endif