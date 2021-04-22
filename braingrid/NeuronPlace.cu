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
        myState->growthInSomas[i] = NULL;
        myState->growthInSomasType[i] = -1;
    }
    
    myState->growthDirection = (BrainGridConstants::Direction)0;
    myState->migrationDest = NULL;
    myState->migrationDestRelativeIdx = -1;
    myState->inboundDendritePlaceID = -1;
    myState->inboundAxonPlaceID = -1;
    myState->travelingDendrite = NULL;
    myState->travelingSynapse = NULL;
}

MASS_FUNCTION int NeuronPlace::getDendriteSpawnTime() {
    return myState->dendriteSpawnTime[myState->dendritesToSpawn - 1];
}

MASS_FUNCTION int NeuronPlace::getDendritesToSpawn() {
    return myState->dendritesToSpawn;
}

MASS_FUNCTION NeuronPlace* NeuronPlace::getMigrationDest() {
    return myState->migrationDest;
}

MASS_FUNCTION int NeuronPlace::getMigrationDestRelIdx() {
    return myState->migrationDestRelativeIdx;
}

MASS_FUNCTION int NeuronPlace::getType() {
    return myState->myType;
}

MASS_FUNCTION int NeuronPlace::getCurIter() {
    return myState->curIter;
}

MASS_FUNCTION int NeuronPlace::getAxonSpawnTime() {
    return myState->axonSpawnTime;
}

MASS_FUNCTION void NeuronPlace::reduceDendritesToSpawn(int amount) {
    myState->dendritesToSpawn -= amount;
}

MASS_FUNCTION GrowingEnd* NeuronPlace::getTravelingSynapse() {
    return myState->travelingSynapse;
}

MASS_FUNCTION GrowingEnd* NeuronPlace::getTravelingDendrite() {
    return myState->travelingDendrite;
}

MASS_FUNCTION void NeuronPlace::setTravelingSynapse(GrowingEnd* gEnd) {
    myState->travelingSynapse = gEnd;
}

MASS_FUNCTION void NeuronPlace::setTravelingDendrite(GrowingEnd* gEnd) {
    myState->travelingDendrite = gEnd;
}

MASS_FUNCTION bool NeuronPlace::isOccupied() {
    return myState->occupied;
}

MASS_FUNCTION void NeuronPlace::setBranchedSynapseSoma(NeuronPlace* branchedSoma) {
    myState->branchedSynapseSoma = branchedSoma;
}

MASS_FUNCTION void NeuronPlace::setBranchedSynapseSomaIdx(int idx) {
    myState->branchedSynapseSomaIdx = idx;
}

MASS_FUNCTION void NeuronPlace::setBranchedDendriteSoma(NeuronPlace* branchedSoma) {
    myState->branchedDendriteSoma = branchedSoma;
}

MASS_FUNCTION void NeuronPlace::setBranchedDendriteSomaIdx(int idx) {
    myState->branchedDendriteSomaIdx = idx;
}

MASS_FUNCTION NeuronPlace* NeuronPlace::getBranchedSynapseSoma() {
    return myState->branchedSynapseSoma;
}

MASS_FUNCTION int NeuronPlace::getBranchedSynapseSomaIdx() {
    return myState->branchedSynapseSomaIdx;
}

MASS_FUNCTION NeuronPlace* NeuronPlace::getBranchedDendriteSoma() {
    return myState->branchedDendriteSoma;
}

MASS_FUNCTION int NeuronPlace::getBranchedDendriteSomaIdx() {
    return myState->branchedDendriteSomaIdx;
}

MASS_FUNCTION double NeuronPlace::getOutputSignal() {
    return myState->outputSignal;
}

MASS_FUNCTION void NeuronPlace::setSimulationTime(int* maxTime) {
    myState->totalIters = maxTime[0];
    myState->curIter = 0;
}

MASS_FUNCTION void NeuronPlace::setNeuronPlaceType(bool* isLoc) {
    if (!isLoc[getIndex()]) {
        myState->myType = BrainGridConstants::EMPTY;
    } else {
        myState->myType = BrainGridConstants::SOMA;
    }
}

MASS_FUNCTION void NeuronPlace::setNeuronSignalType(int* randNum) {
    if (myState->myType == BrainGridConstants::SOMA) {
        int score = randNum[getIndex()] % 30;
        if (score < BrainGridConstants::EXCITATORY) {
            myState->signalType = 0;
        } else if (score < (BrainGridConstants::INHIBITORY + BrainGridConstants::EXCITATORY)) {
            myState->signalType = 1;
        } else {
            myState->signalType = 2;
        }
    }
}

MASS_FUNCTION void NeuronPlace::setGrowthDirections(int* randNums) {
    if (myState->myType == BrainGridConstants::SOMA) {
        myState->growthDirection = (BrainGridConstants::Direction) (randNums[getIndex()] % MAX_NEIGHBORS);
    }
}

MASS_FUNCTION void NeuronPlace::setSpawnTimes(int* randNums) {
    if (myState->myType == BrainGridConstants::SOMA) {
        int numIdx = getIndex() * MAX_NEIGHBORS;
        int numDend = 0;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->neighbors[i] != NULL && ((NeuronPlace*)myState->neighbors[i])->getType() == BrainGridConstants::EMPTY) {
                numDend++;
            }
        }

        // reduce by one for Axon
        myState -> dendritesToSpawn = numDend - 1;
        myState->axonSpawnTime = randNums[numIdx++] % myState->totalIters;
        for (int i = 0; i < myState->dendritesToSpawn; ++i) {
            myState->dendriteSpawnTime[i] = randNums[i + numIdx++] % myState->totalIters;
        }

        // Bubble Sort fastest for small array; sort order descending
        for (int i = 0; i < numDend; ++i) {
            bool swaps = false;
            //when the current item is bigger than next
            for(int j = 0; j < numDend - i - 1; j++) {
                if(myState->dendriteSpawnTime[j] < myState->dendriteSpawnTime[j+1]) {
                    int temp;
                    temp = myState->dendriteSpawnTime[j];
                    myState->dendriteSpawnTime[j] = myState->dendriteSpawnTime[j+1];
                    myState->dendriteSpawnTime[j+1] = temp;
                    swaps = true;
                }
            }     
            
            if(!swaps) {
                break;       // No swap in this pass, so array is sorted
            }    
        }
    }
}

// Executed once during simulation set-up
MASS_FUNCTION void NeuronPlace::findAxonGrowthDestinationFromSoma() {
    if (myState->myType == BrainGridConstants::SOMA) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState ->  migrationDestRelativeIdx = -1;

        NeuronPlace* tmpPlace = (NeuronPlace*)myState->neighbors[myState->growthDirection];
        while (tmpPlace == NULL || tmpPlace->myState->myType != BrainGridConstants::EMPTY || tmpPlace->myState->travelingDendrite != NULL) {
            myState->growthDirection = (BrainGridConstants::Direction) (myState->growthDirection + 1 % N_DESTINATIONS);
            tmpPlace = (NeuronPlace*)myState->neighbors[myState->growthDirection];
        }

        myState->dendritesToSpawn = 0;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->neighbors[i] != NULL || ((NeuronPlace*)myState->neighbors[i])->getType() != BrainGridConstants::SOMA && myState->growthDirection != i) {
                myState->dendritesToSpawn++;
            }
        }

        myState -> migrationDest = (NeuronPlace*)myState -> neighbors[myState -> growthDirection];
        myState -> migrationDestRelativeIdx = myState -> growthDirection;
    }
}

// TODO: Check for correctness
MASS_FUNCTION void NeuronPlace::findDendriteGrowthDestinationFromSoma() {
    if (myState -> myType == BrainGridConstants::SOMA && getAgentPopulation() > 0 && myState -> migrationDest == NULL) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState ->  migrationDestRelativeIdx = -1;
        myState -> dendritesToSpawn--;

        int pos = myState->growthDirection;
        int startPos = pos;
        NeuronPlace* tmpPlace = (NeuronPlace*)myState->neighbors[pos];
        // TODO: Need to protect from having Dendrite go to same place as Axon 
        while (tmpPlace == NULL || tmpPlace->myState->myType != BrainGridConstants::EMPTY || tmpPlace->myState->travelingDendrite != NULL) {
            pos = (pos + 1) % N_DESTINATIONS;
            myState->growthDirection = (BrainGridConstants::Direction)pos;
            tmpPlace = (NeuronPlace*)myState->neighbors[pos];
            if (startPos == pos) {
                return;
            }
        }

        // Set migration destination
        myState->migrationDest = (NeuronPlace*)myState->neighbors[myState->growthDirection];
        myState->migrationDestRelativeIdx = myState->growthDirection;
    }
}

MASS_FUNCTION void NeuronPlace::setNeuronPlaceGrowths() {
    if (myState->myType == BrainGridConstants::EMPTY && !(myState->occupied)) {
        int count = 0;
        GrowingEnd* tmpEnd = (GrowingEnd*)myState->agents[count];
        while (tmpEnd != NULL) {
            if (tmpEnd->getType() == BrainGridConstants::SYNAPSE) {
                if (myState->travelingSynapse == NULL) {
                    myState->travelingSynapse = tmpEnd;
                } else if (tmpEnd->getIndex() < myState->travelingSynapse->getIndex()) {
                    myState->travelingSynapse->terminateAgent();
                    myState->travelingSynapse = tmpEnd;
                }
            }

            else if (tmpEnd->getType() == BrainGridConstants::DENDRITE){
                if (myState->travelingDendrite == NULL) {
                    myState->travelingDendrite = tmpEnd;
                } else if (tmpEnd->getIndex() < myState->travelingDendrite->getIndex()) {
                    myState->travelingDendrite->terminateAgent();
                    myState->travelingDendrite = tmpEnd;
                }
            }

            tmpEnd = (GrowingEnd*)myState->agents[++count];
        }
    }
}

// TODO: Does this need to be updated for 2 growthDestinations, or for 1?
MASS_FUNCTION void NeuronPlace::findGrowthDestinationOutsideSoma() {
    if (myState->myType == BrainGridConstants::EMPTY && !(myState->occupied)) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState ->  migrationDestRelativeIdx = -1;

        NeuronPlace* tmpPlace = (NeuronPlace*)myState->neighbors[myState->growthDirection];
        if (tmpPlace != NULL) {
            myState->migrationDest = tmpPlace;
            myState->migrationDestRelativeIdx = myState->growthDirection;
        }
    }
}

// TODO: Need to handle Axons differently
// Chooses lowest ranked dendrites and synapses and makes connection if possible
// Run after agents->manageAll()
MASS_FUNCTION void NeuronPlace::makeGrowingEndConnections() {
    if (myState->myType != BrainGridConstants::SOMA && !(myState->occupied)) {
        GrowingEnd* incomingDendrite = NULL;
        GrowingEnd* incomingSynapse = NULL;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->agents[i] != NULL) {
                if (((GrowingEnd*)myState->agents[i])->getType() == BrainGridConstants::SYNAPSE) {
                    if (incomingSynapse == NULL) {
                        incomingSynapse = (GrowingEnd*)(myState->agents[i]);
                    } else if (((GrowingEnd*)myState->agents[i])->getIndex() < incomingSynapse->getIndex()) {
                        incomingSynapse->terminateAgent();
                        incomingSynapse = (GrowingEnd*)myState->agents[i];
                    } else {
                        myState->agents[i]->terminateAgent();
                    }
                }
                // Dendrite
                else if (((GrowingEnd*)myState->agents[i])->getType() == BrainGridConstants::DENDRITE) {
                    if (incomingDendrite == NULL) {
                        incomingDendrite = (GrowingEnd*)myState->agents[i];
                    } else if (((GrowingEnd*)myState->agents[i])->getIndex() < incomingDendrite->getIndex()) {
                        incomingDendrite->terminateAgent();
                        incomingDendrite = (GrowingEnd*)myState->agents[i];
                    } else {
                        myState->agents[i]->terminateAgent();
                    }
                }
            }
        }


        if (myState->travelingDendrite != NULL && myState->travelingSynapse != NULL) {
            if (myState->travelingDendrite->getSomaIndex() != myState->travelingSynapse->getSomaIndex()) {
                myState->occupied = true;
                myState->travelingDendrite->setGrowing(false);
                myState->travelingSynapse->setGrowing(false);
                myState->travelingSynapse->setConnectPlace(myState->travelingDendrite->getSoma());
                myState->travelingSynapse->setConnectPlaceIndex(myState->travelingDendrite->getSomaIndex());    
            }

            myState->travelingDendrite->terminateAgent();
        }
    }
}

MASS_FUNCTION void NeuronPlace::createSignal(int* randNums) {
    if (myState->myType == BrainGridConstants::SOMA) {
        myState->outputSignal = 0.0;
        switch(myState->signalType) {
            case 0:
                myState->outputSignal = (randNums[getIndex()] % 100 + 1 <= 
                BrainGridConstants::ACTIVATING_SIGNAL ) ? (1.0) : (0.0);
                myState->outputSignal += myState->inputSignal * (1 + BrainGridConstants::SIGNAL_MODULATION);
                break;
            case 1:
                myState->outputSignal += myState->inputSignal * BrainGridConstants::SIGNAL_MODULATION;
                break;
            case 2:
                myState->outputSignal = myState->inputSignal;
                break;
        }
    }
}

// TODO: needs to use agents array to get signals from migrating Agents
MASS_FUNCTION void NeuronPlace::processSignals() {
    if (myState->myType == BrainGridConstants::SOMA) {
        myState->inputSignal = 0.0;
        for (int i = 0; i < N_DESTINATIONS; ++i) {
            if (((GrowingEnd*)myState->agents[i])->getType() == BrainGridConstants::SYNAPSE && 
                    ((GrowingEnd*)myState->agents[i])->getConnectPlace() != NULL) {
                // accumulate signal
                myState->inputSignal += ((GrowingEnd*)myState->agents[i])->getSignal();
            }
        }
    }
}

MASS_FUNCTION void NeuronPlace::updateIters() {
    
}

MASS_FUNCTION void NeuronPlace::callMethod(int functionId, void *argument) {
	switch (functionId) {
        case SET_TIME:
            setSimulationTime((int*)argument);
        case INIT_NEURONS:
            setNeuronPlaceType((bool*)argument);
            break;
        case SET_NEURON_SIGNAL_TYPE:
            setNeuronSignalType((int*)argument);
            break;
        case SET_SPAWN_TIMES:
            setSpawnTimes((int*)argument);
            break;
        case SET_GROWTH_DIRECTIONS:
            setGrowthDirections((int*)argument);
            break;
        case FIND_AXON_GROWTH_DESTINATIONS_FROM_SOMA:
            findAxonGrowthDestinationFromSoma();
            break;
        case FIND_DENDRITE_GROWTH_DESTINATIONS_FROM_SOMA:
            findDendriteGrowthDestinationFromSoma();
            break;
        case SET_NEURON_PLACE_MIGRATIONS:
            setNeuronPlaceGrowths();
            break;
        case FIND_GROWTH_DESTINATIONS_OUTSIDE_SOMA:
            findGrowthDestinationOutsideSoma();
            break;    
        case MAKE_CONNECTIONS:
            makeGrowingEndConnections();
            break;
        case CREATE_SIGNAL:
            createSignal((int*)argument);
            break;
        case PROCESS_SIGNALS:
            processSignals();
            break;
        case UPDATE_ITERS:
            updateIters();
            break;
        default:
			break;
    }
}
