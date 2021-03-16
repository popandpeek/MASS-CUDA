#include <ctime> // clock_t,
#include <stdio.h> //remove after debugging
#include <cuda_runtime.h>
#include "NeuronPlace.h"
#include "BrainGridConstants.h"
#include "../src/settings.h"

using namespace mass;


MASS_FUNCTION NeuronPlace::NeuronPlace(PlaceState *state, void *argument) :
		Place(state, argument) {

	myState = (NeuronPlaceState*) state;
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        growthIn[i] = -1;
    }
    
    myState->migrationDest = NULL;
    inboundSynapsePlaceID = -1;
    inboundAxonPlaceID = -1;
}

MASS_FUNCTION void NeuronPlace::callMethod(int functionId, void *argument) {
	switch (functionId) {
        case SET_TIME:
            setSimulationTime(*(int*) argument);
        case INIT_NEURONS:
            setNeuronPlaceType((bool*)argumentl);
            break;
        case SET_NEURON_SIGNAL_TYPE:
            setActiveNeuronParams((int*)argument);
            break;
        case COUNT_SOMAS:
            countSomas((int*)argument);
            break;
        case SET_SPAWN_TIMES:
            setSpawnTimes((int*)argument);
            break;
        case FIND_AXON_GROWTH_DESTINATIONS_FROM_SOMA:
            findAxonGrowthDestinationFromSoma();
            break;
        case FIND_DENDRITE_GROWTH_DESTINATIONS_FROM_SOMA:
            findDendriteGrowthDestination();
            break;
        case SET_BRANCH_MIGRATION_DESTINATIONS:
            setSpawnedBranchMigrationDestinations();
            break;
        case MAKE_CONNECTIONS:
            makeGrowingEndConnections();
            break;
        default:
			break;
    }
}

MASS_FUNCTION void NeuronPlace::setSimulationTime(int maxTime) {
    myState->totalIters = maxTime;
    myState->curIter = 0;
}

MASS_FUNCTION void NeuronPlace::setNeuronPlaceType(bool* isLoc) {
    if (!isLoc[getIndex()]) {
        myState->myType = EMPTY;
    } else {
        myState->myType = SOMA;
    }
}

MASS_FUNCTION void setNeuronSignalType(int randNum) {
    if (myState->myType == SOMA) {
        int score = randNum[getIndex()] % 30;
        if (score < BrainGridConstants::EXCITATORY) {
            signal = BrainGridConstants::EXCITATORY;
        } else if (score < (BrainGridConstants::INHIBITORY + BrainGridConstants::EXCITATORY)) {
            signal = BrainGridConstants::INHIBITORY;
        } else {
            signal = BrainGridConstants::NEUTRAL;
        }
}
}

MASS_FUNCTION void NeuronPlace::countSomas(int* count) {
    if (myState->myType != BrainGridConstants::EMPTY) {
        atomicAdd(count, 1);
    }
}

MASS_FUNCTION void setSpawnTimes(int* randNums) {
    if (myState->myType == BrainGridConstants::SOMA) {
        int numIdx = getIndex() * MAX_NEIGHBORS;
        int numDend = 0;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->neighbors[i] != NULL && ((NeuronPlace*)myState->neighbors[i])->myState->myType == EMPTY) {
                numDend++;
            }
        }

        // reduce by one for Axon
        myState -> dendritesToSpawn = numDend - 1;
        myState->axonSpawnTime = randNums[numIdx++] % totalIters;
        for (int i = 0; i < myState->dendritesToSpawn; ++i) {
            myState->dendriteSpawnTime[i] = randNums[i + numIdx++] % totalIters;
        }

        // Bubble Sort fastest for small array
        for (int i = 0; i < numDend; ++i) {
            int swaps = 0;
            //when the current item is bigger than next
            for(int j = 0; j < numDend - i - 1; j++) {
                if(myState->dendriteSpawnTime[j] > dendriteSpawnTime[j+1]) {
                    int temp;
                    temp = myState->dendriteSpawnTime[j];
                    myState->dendriteSpawnTime[j] = myState->dendriteSpawnTime[j+1];
                    myState->dendriteSpawnTime[j+1] = temp;
                    swaps = 1;
                }
            }     
            
            if(!swaps) {
                break;       // No swap in this pass, so array is sorted
            }    
        }
    }
}

MASS_FUNCTION void setGrowthDirection(int* randNums) {
    myState->growthDirection = randNums[getIndex()] % N_DESTINATIONS;
}

// Executed once during simulation set-up
MASS_FUNCTION void NeuronPlace::findAxonGrowthDestinationFromSoma() {
    if (myState->myType == SOMA) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState ->  migrationDestRelativeIdx = -1;

        Place* tmpPlace = myState->neighbors[myState->axonGrowthDirection];
        while (tmpPlace == NULL || tmpPlace->myType != EMPTY || tmpPlace->inboundAxonPlaceID >= 0) {
            myState->growthDirection = myState->axonGrowthDirection + 1 % N_DESTINATIONS;
            tmpPlace = myState->neighbors[myState->axonGrowthDirection];
        }

        myState->dendritesToSpawn = 0;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->neighbors[i] != NULL && myState->axonGrowthDirection != i) {
                myState->dendritesToSpawn++;
            }
        }

        // Set neighbor as having growth in from this Place
        tmpPlace -> inboundAxonPlaceID = getIndex();
        myState -> migrationDest = (NeuronPlace*)myState -> neighbors[myState -> axonGrowthDirection];
        myState -> migrationDestRelativeIdx = myState -> axonGrowthDirection;
    }
}

MASS_FUNCTION void NeuronPlace::findDendriteGrowthDestinationFromSoma() {
    if (myState -> myType == SOMA && getAgentPopulation() > 0 && myState -> migrationDest == NULL) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState ->  migrationDestRelativeIdx = -1;
        myState -> dendritesToSpawn--;

        int pos = myState->dendriteGrowthDirection;
        int startPos = pos;
        Place* tmpPlace = myState->neighbors[pos];
        while (tmpPlace == NULL || tmpPlace->inboundDendritePlaceID >= 0 || 
                    tmpPlace->myType != EMPTY || tmpPlace->travelingDendrite != NULL) {
            pos = (pos + 1) % N_DESTINATIONS;
            myState->dendriteGrowthDirection = pos;
            tmpPlace = myState->neighbors[pos];
            if (startPos == pos) {
                return;
            }
        }

        // Set neighbor as having growth in from this Place
        tmpPlace -> inboundDendritePlaceID = getIndex()
        myState -> migrationDest = (NeuronPlace*)myState-> neighbors[myState->dendriteGrowthDirection];
        myState -> migrationDestRelativeIdx = myState->dendriteGrowthDirection;
    }
}

MASS_FUNCTION void NeuronPlace::setSpawnedBranchMigrationDestinations() {
    if (myState->myType == EMPTY && !(myState->occupied)) {
        for (int i = 0; i < MAX_AGENTS; ++i) {
            // If Agent just spawned
            if (agents[i] != NULL && ((GrowingEnd*)agents[i])->isGrowing && 
                    ((GrowingEnd*)agents[i])->myType != AXON && !((GrowingEnd*)agents[i])->justGrew) {
                for (int j = 0; j < MAX_NEIGHBORS; ++j) {
                    if (myState->neighbors[j]->growthInSomas[(j + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] != NULL && 
                            growthInSomasType[j] == ((GrowingEnd*)agents[i])->myType) {
                        ((GrowingEnd*)agents[i])->growthDirection = j;
                        ((GrowingEnd*)agents[i])->mySoma = growthInSomas[j];
                        ((GrowingEnd*)agents[i])->mySomaIndex = growthInSomas[j]->getIndex();
                        myState->neighbors[j]->growthInSomas[j] = NULL;
                        myState->neighbors[j]->growthInSomasType[j] = -1;
                    }
                }
            }
        }
    }
}

// Chooses lowest ranked dendrites and synapses and makes connection if possible
// Allows Axon's to travel across (?)
// TODO: Do we need to handle Axons differntly? 
// Run after agents->manageAll()
MASS_FUNCTION void NeuronPlace::makeGrowingEndConnections() {
    if (myState->myType != SOMA && !(myState->occupied)) {
        GrowingEnd* incomingDendrite = NULL;
        GrowingEnd* incomingSynapse = NULL;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (potentialNextAgents[i] != NULL) {
                if (potentialNextAgents[i]->myType == SYNAPSE) {
                    if (incomingSynapse == NULL) {
                        incomingSynapse = potentialNextAgents[i];
                    } else if (potentialNextAgents[i]->index < incomingSynapse->index) {
                        incomingSynapse->terminateAgent();
                        incomingSynapse = potentialNextAgents[i];
                    } else {
                        potentialNextAgents[i]->terminateAgent();
                    }
                }
                // Dendrite
                else {
                    if (incomingDendrite == NULL) {
                        incomingDendrite = potentialNextAgents[i];
                    } else if (potentialNextAgents[i]->index < incomingDendrite->index) {
                        incomingDendrite->terminateAgent();
                        incomingDendrite = potentialNextAgents[i];
                    } else {
                        potentialNextAgents[i]->terminateAgent();
                    }
                }
            }
        }


        if (myState->travelingDendrite != NULL && myState->travelingSynapse != NULL) {
            if (myState->travelingDendrite->mySomaIndex != myState->travelingSynapse->mySomaIndex) {
                myState->occupied = true;
                myState->travelingDendrite->isGrowing = false;
                myState->travelingSynapse->isGrowing = false;
            } else {
                myState->travelingDendrite->terminateAgent();
            }
        }
    }
}