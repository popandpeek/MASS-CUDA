#ifndef NEURONSTATE_H
#define NEURONSTATE_H


#include "../src/PlaceState.h"
#include "NeuronPlace.h"
#include "BrainGridConstants.h"


class NeuronPlace;
class GrowingEnd;

class NeuronPlaceState: public mass::PlaceState {
public:

    BrainGridConstants::NPartType myType; // EMPTY, SOMA, AXON, DENDRITE, SYNAPTIC_TERMINAL
    int totalIters;
    int curIter;
    int signalType;
    double inputSignal;
    double outputSignal;
    
    // SOMA behavior params
    unsigned int dendritesToSpawn;
    int axonSpawnTime;
    int dendriteSpawnTime[MAX_NEIGHBORS];
    BrainGridConstants::Direction growthDirection;

    // migration params
    NeuronPlace *migrationDest;
    int migrationDestRelativeIdx;
    NeuronPlace* branchedSynapseSoma;
    NeuronPlace* branchedDendriteSoma;
    int branchedSynapseSomaIdx;
    int branchedDendriteSomaIdx;
    
    // Connections made are moved to connectedNeurons each turn and cleared
    bool occupied;
    bool travelingDendrite;
    bool travelingSynapse;
};

#endif 