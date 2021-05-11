#ifndef NEURONPLACE_H
#define NEURONPLACE_H

#include "../src/Place.h"
#include "../src/Logger.h"
#include "NeuronPlaceState.h"
#include "GrowingEnd.h"

class NeuronPlace: public mass::Place {

public:

    const static int SET_TIME = 0;
    const static int INIT_NEURONS = 1;
    const static int SET_NEURON_SIGNAL_TYPE = 2;
    const static int SET_SPAWN_TIMES = 3;
    const static int SET_GROWTH_DIRECTIONS = 4;
    const static int FIND_AXON_GROWTH_DESTINATIONS_FROM_SOMA = 5;
    const static int FIND_DENDRITE_GROWTH_DESTINATIONS_FROM_SOMA = 6;
    const static int SET_NEURON_PLACE_MIGRATIONS = 7;
    const static int FIND_GROWTH_DESTINATIONS_OUTSIDE_SOMA = 8;
    const static int MAKE_CONNECTIONS = 9;
    const static int CREATE_SIGNAL = 10;
    const static int PROCESS_SIGNALS = 11;
    const static int UPDATE_ITERS = 12;

    MASS_FUNCTION NeuronPlace(mass::PlaceState *state, void *argument = NULL);
	MASS_FUNCTION ~NeuronPlace();

	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual NeuronPlaceState* getState();
    
    MASS_FUNCTION NeuronPlace* getMigrationDest();
    MASS_FUNCTION int getMigrationDestRelIdx();
    MASS_FUNCTION int getGrowthDirection();
    MASS_FUNCTION int getType();
    MASS_FUNCTION int getCurIter();
    MASS_FUNCTION int getDendriteSpawnTime();
    MASS_FUNCTION int getAxonSpawnTime();
    MASS_FUNCTION int getDendritesToSpawn();
    MASS_FUNCTION void reduceDendritesToSpawn(int);
    MASS_FUNCTION bool getTravelingSynapse();
    MASS_FUNCTION bool getTravelingDendrite();
    MASS_FUNCTION void setTravelingDendrite(bool);
    MASS_FUNCTION void setTravelingSynapse(bool);
    MASS_FUNCTION bool isOccupied();
    MASS_FUNCTION void setOccupied(bool);
    MASS_FUNCTION void setBranchedSynapseSoma(NeuronPlace*);
    MASS_FUNCTION void setBranchedSynapseSomaIdx(int);
    MASS_FUNCTION void setBranchedDendriteSoma(NeuronPlace*);
    MASS_FUNCTION void setBranchedDendriteSomaIdx(int);
    MASS_FUNCTION NeuronPlace* getBranchedSynapseSoma();
    MASS_FUNCTION int getBranchedSynapseSomaIdx();
    MASS_FUNCTION NeuronPlace* getBranchedDendriteSoma();
    MASS_FUNCTION int getBranchedDendriteSomaIdx();
    MASS_FUNCTION double getOutputSignal();


private:
 
    NeuronPlaceState* myState;

    MASS_FUNCTION void setSimulationTime(int*);
    MASS_FUNCTION void setNeuronPlaceType(bool*);
    MASS_FUNCTION void setNeuronSignalType(int*);
    MASS_FUNCTION void setActiveNeuronParams(int*);
    MASS_FUNCTION void setSpawnTimes(int*);
    MASS_FUNCTION void setGrowthDirections(int*);
    MASS_FUNCTION void findAxonGrowthDestinationFromSoma();
    MASS_FUNCTION void findDendriteGrowthDestinationFromSoma();
    MASS_FUNCTION void setNeuronPlaceGrowths();
    MASS_FUNCTION void findGrowthDestinationOutsideSoma();
    MASS_FUNCTION void makeGrowingEndConnections();
    MASS_FUNCTION void createSignal(int*);
    MASS_FUNCTION void processSignals();
    MASS_FUNCTION void updateIters();
};

#endif