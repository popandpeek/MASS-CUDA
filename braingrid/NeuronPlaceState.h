#ifndef NEURONSTATE_H
#define NEURONSTATE_H


#include "../src/PlaceState.h"
#include "NeuronPlaceState.h"
#include "BrainGridConstants.h"


class NeuronPlace;

class NeuronPlaceState: public mass::PlaceState {

    BrainGridConstants::NPartType myType; // EMPTY, SOMA, AXON, DENDRITE, SYNAPTIC_TERMINAL
    int totalIters;
    int curIter;
    int signalType;
    double inputSignal;
    double outputSignal;
    
    // SOMA behavior params
    unsigned int dendritesToSpawn;
    int axonSpawnTime;
    int *dendriteSpawnTime[MAX_NEIGHBORS];
    BrainGridConstants::Direction axonGrowthDirection;
    BrainGridConstants::Direction dendriteGrowthDirection;
    
    // stores SOMA locations for spawning GrowingEnds
    NeuronPlace *growthInSomas[MAX_NEIGHBORS];
    int *growthInSomasType[MAX_NEIGHBORS];

    // migration params
    NeuronPlace *migrationDest;
    int migrationDestRelativeIdx;
    
    // Connections made are moved to connectedNeurons each turn and cleared
    bool occupied;
    int inboundAxonPlaceID;
    int inboundDendritePlaceID;
    GrowingEnd* travelingDendrite;
    GrowingEnd* travelingSynapse;
};

#endif 