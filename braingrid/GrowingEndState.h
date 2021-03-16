#ifndef DENDRITESTATE_H
#define DENDRITESTATE_H


#include "../src/AgentState.h"
#include "BrainGridConstants.h"


class GrowingEnd;
class NeuronPlace;

class GrowingEndState: public mass::AgentState {
    
    // NeuronPlace of origin
    NeuronPlace* mySoma;
    int mySomaIndex;
    BrainGridConstants::NPartType myType;
    int spawnTime;
    BrainGridConstants::Direction growthDirection;
    bool hasSpawned;
    bool isGrowing;
    bool justGrew;
    int branchCountRemaining;
    int branchGrowthRemaining;
    int destinationIdx;
};

#endif 