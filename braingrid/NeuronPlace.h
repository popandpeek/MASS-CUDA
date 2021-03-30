#ifndef NEURONPLACE_H
#define NEURONPLACE_H

#include "../src/Place.h"
#include "../src/Logger.h"
#include "NeuronPlaceState.h"

class NeuronPlace: public mass::Place {

public:

    const static int SET_TIME = 0;
    const static int INIT_NEURONS = 1;
    const static int SET_NEURON_SIGNAL_TYPE = 2;
    const static int COUNT_SOMAS = 3;
    const static int SET_SPAWN_TIMES = 4;
    const static int FIND_AXON_GROWTH_DESTINATIONS_FROM_SOMA = 5;
    const static int FIND_DENDRITE_GROWTH_DESTINATIONS_FROM_SOMA = 6;
    const static int SET_BRANCH_MIGRATION_DESTINATIONS = 7;
    const static int MAKE_CONNECTIONS = 8;
    const static int CREATE_SIGNAL = 9;

    MASS_FUNCTION NeuronPlace(mass::PlaceState *state, void *argument = NULL);
	MASS_FUNCTION ~NeuronPlace();

	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    
private:

    NeuronPlaceState* myState;

    MASS_FUNCTION void setSimulationTime(int*);
    MASS_FUNCTION void setNeuronPlaceType(bool*);
    MASS_FUNCTION void setActiveNeuronParams(int*);
    MASS_FUNCTION void countSomas(int*);
    MASS_FUNCTION void setSpawnTimes(int*);
    MASS_FUNCTION void findAxonGrowthDestinationFromSoma();
    MASS_FUNCTION void findDendriteGrowthDestination();
    MASS_FUNCTION void setSpawnedBranchMigrationDestinations();
    MASS_FUNCTION void makeGrowingEndConnections();
    MASS_FUNCTION void createSignal(int*);

};

#endif