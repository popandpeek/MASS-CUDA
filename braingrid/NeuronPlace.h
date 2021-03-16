#ifndef NEURONPLACE_H
#define NEURONPLACE_H

#include "../src/Place.h"
#include "../src/Logger.h"
#include "NeuronPlaceState.h"

class NeuronPlace: public mass::Place {

public:

    const static int INIT_NEURONS = 0;
    const static int SET_NEURON_SIGNAL_TYPE = 1;
    const static int COUNT_SOMAS = 2;
    const static int FIND_MIGRATION_DESTINATION = 3;
    const static int FIND_AXON_GROWTH_DIRECTION = 4;

    MASS_FUNCTION NeuronPlace(mass::PlaceState *state, void *argument = NULL);
	MASS_FUNCTION ~NeuronPlace();

	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    
private:

    NeuronPlaceState* myState;

    MASS_FUNCTION void setNeuronPlaceType(int t);
    MASS_FUNCTION void countSomas(int* count);
    MASS_FUNCTION void findMigrationDestination()''


};

#endif