#ifndef NEURONSTATE_H
#define NEURONSTATE_H


#include "../src/PlaceState.h"
#include "NeuronPlace.h"
#include "BrainGridConstants.h"


class NeuronPlace;
class GrowingEnd;

class NeuronPlaceState: public mass::PlaceState {
public:

    // BrainGridConstants::NPartType myType; // EMPTY, SOMA, AXON, DENDRITE, SYNAPTIC_TERMINAL
    int myPlaceType;
    int totalIters;
    int curIter;
    int signalType;
    double inputSignal;
    double outputSignal;
    
    // SOMA behavior params
    int dendritesToSpawn;
    int axonSpawnTime;
    int dendriteSpawnTime[MAX_NEIGHBORS - 1];
    // BrainGridConstants::Direction growthDirection;
    int axonGrowthDirection;
    int growthDirection;

    // migration params
    NeuronPlace *migrationDest;
    int migrationDestRelativeIdx;
    NeuronPlace *branchMigrationDest;
    int branchMigrationDestRelativeIdx;
    
    NeuronPlace* branchedSynapseSoma;
    NeuronPlace* branchedDendriteSoma;
    int branchedSynapseSomaIdx;
    int branchedDendriteSomaIdx;
    bool occupied;
};

#endif 