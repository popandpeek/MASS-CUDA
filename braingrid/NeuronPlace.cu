#include <ctime> // clock_t,
#include <stdio.h> //remove after debugging
#include <cuda_runtime.h>
#include "NeuronPlace.h"
#include "../src/settings.h"

using namespace mass;


MASS_FUNCTION NeuronPlace::NeuronPlace(PlaceState *state, void *argument) :
		Place(state, argument) {

	myState = (NeuronPlaceState*) state;
    
    myState->myPlaceType = BrainGridConstants::EMPTY;
    myState->growthDirection = BrainGridConstants::NORTH;
    myState->migrationDest = NULL;
    myState->migrationDestRelativeIdx = -1;
    myState->dendritesToSpawn = 0;
    myState->branchMigrationDest = NULL;
    myState->branchMigrationDestRelativeIdx = -1;
    myState->branchedSynapseSoma = NULL;
    myState->branchedDendriteSoma = NULL;
}

MASS_FUNCTION void NeuronPlace::setSimulationTime(int* maxTime) {
    myState->totalIters = maxTime[getDevIndex()];
    myState->curIter = 0;
}

MASS_FUNCTION void NeuronPlace::setNeuronPlaceType(int* isLoc) {
    if (isLoc[getDevIndex()] == BrainGridConstants::SOMA) {
        myState->myPlaceType = BrainGridConstants::SOMA;
    }
}

MASS_FUNCTION void NeuronPlace::setNeuronSignalType(int* randNum) {
    if (myState->myPlaceType == BrainGridConstants::SOMA) {
        int score = randNum[getDevIndex()] % 30;
        if (score < BrainGridConstants::EXCITATORY) {
            myState->signalType = BrainGridConstants::E;
        } else if (score < (BrainGridConstants::INHIBITORY + BrainGridConstants::EXCITATORY)) {
            myState->signalType = BrainGridConstants::I;
        } else {
            myState->signalType = BrainGridConstants::N;
        }
    }
}

MASS_FUNCTION void NeuronPlace::setGrowthDirections(int* randNums) {
    if (myState->myPlaceType == BrainGridConstants::SOMA) {
        myState->growthDirection = abs(randNums[getDevIndex()]) % MAX_NEIGHBORS;
    }
}

MASS_FUNCTION void NeuronPlace::setSpawnTimes(int* randNums) {
    if (myState->myPlaceType == BrainGridConstants::SOMA) {
        int numIdx = getDevIndex() * MAX_NEIGHBORS;
        int numBranches = 0;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->neighbors[i] != NULL && ((NeuronPlace*)myState->neighbors[i])->getPlaceType() == BrainGridConstants::EMPTY) {
                numBranches++;
            }
        }

        if (!(numBranches > 0)) {
            return;
        }

        // reduce by one for Axon        
        myState->axonSpawnTime = randNums[numIdx];
        myState -> dendritesToSpawn = numBranches - 1;
        numIdx += 1;
        for (int i = 0; i < myState->dendritesToSpawn; ++i) {
            myState->dendriteSpawnTime[i] = randNums[i + numIdx];
        }

        // Bubble Sort OK for small array; sort order descending
        for (int i = 0; i < (myState->dendritesToSpawn - 1); ++i) {
            bool swaps = false;
            //when the current item is bigger than next
            for(int j = 0; j < myState->dendritesToSpawn - i - 1; j++) {
                if(myState->dendriteSpawnTime[j] > myState->dendriteSpawnTime[j+1]) {
                    int temp = myState->dendriteSpawnTime[j];
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
    if (myState->myPlaceType == BrainGridConstants::SOMA) {
        //initially assume we won't find a suitable place
        myState -> migrationDest = NULL; 
        myState -> migrationDestRelativeIdx = -1;

        NeuronPlace* tmpPlace = (NeuronPlace*)myState->neighbors[myState->growthDirection];
        int count = 0;
        while (tmpPlace == NULL || tmpPlace->getPlaceType() == BrainGridConstants::SOMA) {
            myState->growthDirection = (myState->growthDirection + 1) % MAX_NEIGHBORS;
            tmpPlace = (NeuronPlace*)myState->neighbors[myState->growthDirection];
            if (count == MAX_NEIGHBORS) {
                return;
            }
            count++;
        }

        myState->axonGrowthDirection = myState->growthDirection;
        myState -> migrationDest = (NeuronPlace*)(myState->neighbors[myState->growthDirection]);
        myState -> migrationDestRelativeIdx = myState->growthDirection;
    }
}

// TODO: Check for correctness
MASS_FUNCTION void NeuronPlace::findDendriteGrowthDestinationFromSoma() {
    if (myState -> myPlaceType == BrainGridConstants::SOMA && getAgentPopulation() > 0) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState -> migrationDestRelativeIdx = -1;

        int pos = myState->growthDirection;
        int startPos = pos;
        NeuronPlace* tmpPlace = (NeuronPlace*)myState->neighbors[pos];
        while (tmpPlace == NULL || tmpPlace->getPlaceType() != BrainGridConstants::EMPTY || 
                pos == myState->axonGrowthDirection) {
            pos = (pos + 1) % N_DESTINATIONS;
            myState->growthDirection = pos;
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

// TODO: Does this need to be updated for 2 growthDestinations, or for 1?
MASS_FUNCTION void NeuronPlace::findGrowthDestinationOutsideSoma() {
    if (myState->myPlaceType == BrainGridConstants::EMPTY && getAgentPopulation() > 0 && myState->occupied == false) {
        myState -> migrationDest = NULL; //initially assume we won't find a suitable place
        myState -> migrationDestRelativeIdx = -1;

        GrowingEnd* tmpAgent = NULL;
        for (int i = 0; i < MAX_AGENTS; ++i) {
            if (myState->agents[i] != NULL) {
                tmpAgent = (GrowingEnd*)myState->agents[i];
                myState->growthDirection = tmpAgent->getGrowthDirection();
                break;
            }
        }

        if (tmpAgent == NULL) {
            return;
        }

        NeuronPlace* tmpPlace = ((NeuronPlace*)myState->neighbors[myState->growthDirection]);
        if (tmpPlace == NULL || tmpPlace->getPlaceType() != BrainGridConstants::EMPTY || tmpPlace->isOccupied()) {
            // Axons may grow +/- 45 degrees
            if (tmpAgent->getAgentType() == BrainGridConstants::AXON) {
                int leftGrowthDir = (myState->growthDirection - 1) % MAX_NEIGHBORS;
                int rightGrowthDir = (myState->growthDirection + 1) % MAX_NEIGHBORS;
                NeuronPlace* tmpPlaceLeft = ((NeuronPlace*)myState->neighbors[leftGrowthDir]);
                NeuronPlace* tmpPlaceRight = ((NeuronPlace*)myState->neighbors[rightGrowthDir]);
                if (tmpPlaceLeft != NULL && tmpPlaceLeft->getPlaceType() == BrainGridConstants::EMPTY && 
                        tmpPlaceLeft->isOccupied() == false) {
                    tmpPlace = tmpPlaceLeft;
                    myState->growthDirection = leftGrowthDir;
                } else if (tmpPlaceRight != NULL && tmpPlaceRight->getPlaceType() == BrainGridConstants::EMPTY && 
                        tmpPlaceRight->isOccupied() == false){
                    tmpPlace = tmpPlaceRight;
                    myState->growthDirection = rightGrowthDir;
                } else {
                    return;
                }
            } else {
                return;
            }
        }

        myState->migrationDest = tmpPlace;
        myState->migrationDestRelativeIdx = myState->growthDirection;
    }
}

MASS_FUNCTION void NeuronPlace::findBranchDestinationsOutsideSoma() {
    if (myState->myPlaceType == BrainGridConstants::EMPTY && myState->occupied == false && 
            getAgentPopulation() > 0) {
        myState->branchMigrationDest = NULL;
        myState->branchMigrationDestRelativeIdx = -1;

        int tempIdx1 = (myState->growthDirection - 1) % MAX_NEIGHBORS;
        NeuronPlace* tmpPlace1 = (NeuronPlace*)myState->neighbors[tempIdx1];
        int tempIdx2 = (myState->growthDirection + 1) % MAX_NEIGHBORS;
        NeuronPlace* tmpPlace2 = (NeuronPlace*)myState->neighbors[tempIdx2];
        if (tmpPlace1 != NULL && tmpPlace1->getPlaceType() == BrainGridConstants::EMPTY 
                && tmpPlace1->isOccupied() == false) {
            myState->branchMigrationDest = tmpPlace1;
            myState->branchMigrationDestRelativeIdx = tempIdx1;
        }
        else if (tmpPlace2 != NULL && tmpPlace2->getPlaceType() == BrainGridConstants::EMPTY 
                && tmpPlace2->isOccupied() == false) {
            myState->branchMigrationDest = tmpPlace2;
            myState->branchMigrationDestRelativeIdx = tempIdx2; 
        }
    }
}

// Chooses lowest ranked dendrites and synapses and makes connection if possible
// Run after agents->manageAll()
MASS_FUNCTION void NeuronPlace::makeGrowingEndConnections() {
    if (myState->myPlaceType == BrainGridConstants::EMPTY && myState->occupied == false && getAgentPopulation() > 0) {
        GrowingEnd* incomingDendrite = NULL;
        GrowingEnd* incomingSynapse = NULL;
        for (int i = 0; i < MAX_NEIGHBORS; ++i) {
            if (myState->agents[i] != NULL) {
                if (((GrowingEnd*)myState->agents[i])->getAgentType() == BrainGridConstants::SYNAPSE) {
                    if (incomingSynapse == NULL) {
                        incomingSynapse = (GrowingEnd*)(myState->agents[i]);
                    } else if (((GrowingEnd*)myState->agents[i])->getIndex() < incomingSynapse->getIndex()) {
                        incomingSynapse->resetGrowingEnd();
                        incomingSynapse->markAgentForTermination();
                        incomingSynapse = (GrowingEnd*)myState->agents[i];
                    } else {
                        ((GrowingEnd*)myState->agents[i])->resetGrowingEnd();
                        ((GrowingEnd*)myState->agents[i])->markAgentForTermination();
                    }
                }
                // Dendrite
                else if (((GrowingEnd*)myState->agents[i])->getAgentType() == BrainGridConstants::DENDRITE) {
                    if (incomingDendrite == NULL) {
                        incomingDendrite = (GrowingEnd*)myState->agents[i];
                    } else if (((GrowingEnd*)myState->agents[i])->getIndex() < incomingDendrite->getIndex()) {
                        incomingDendrite->resetGrowingEnd();
                        incomingDendrite->markAgentForTermination();
                        incomingDendrite = (GrowingEnd*)myState->agents[i];
                    } else {
                        ((GrowingEnd*)myState->agents[i])->resetGrowingEnd();
                        ((GrowingEnd*)myState->agents[i])->markAgentForTermination();;
                    }
                }
            }
        }


        if (incomingSynapse != NULL && incomingDendrite != NULL) {
            if (incomingSynapse->getSomaIndex() != incomingDendrite->getSomaIndex()) {
                setOccupied(true);
                incomingSynapse->setGrowing(false);
                incomingSynapse->setConnectPlace(incomingDendrite->getSoma());
                incomingSynapse->setConnectPlaceIndex(incomingDendrite->getSomaIndex());     
            }
            incomingDendrite->resetGrowingEnd();
            incomingDendrite->markAgentForTermination();
        }
    }
}

MASS_FUNCTION void NeuronPlace::createSignal(int* randNums) {
    if (myState->myPlaceType == BrainGridConstants::SOMA) {
        myState->outputSignal = 0.0;
        switch(myState->signalType) {
            case 0:
                myState->outputSignal = (randNums[getDevIndex()] % 100 + 1 <= 
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
    if (myState->myPlaceType == BrainGridConstants::SOMA) {
        myState->inputSignal = 0.0;
        for (int i = 0; i < N_DESTINATIONS; ++i) {
            if (myState->agents[i] != NULL && ((GrowingEnd*)myState->agents[i])->getAgentType() == BrainGridConstants::SYNAPSE && 
                    ((GrowingEnd*)myState->agents[i])->getConnectPlace() != NULL) {
                // accumulate signal
                myState->inputSignal += ((GrowingEnd*)myState->agents[i])->getSignal();
            }
        }
    }
}

MASS_FUNCTION void NeuronPlace::updateIters() {
    myState->curIter++;
}

MASS_FUNCTION void NeuronPlace::removeMarkedAgentsFromPlace() {
    removeMarkedAgents();
}

MASS_FUNCTION NeuronPlace::~NeuronPlace() {
	// nothing to delete
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

MASS_FUNCTION NeuronPlace* NeuronPlace::getBranchMigrationDest() {
    return myState->branchMigrationDest;
}

MASS_FUNCTION int NeuronPlace::getBranchMigrationDestRelIdx() {
    return myState->branchMigrationDestRelativeIdx;
}

MASS_FUNCTION int NeuronPlace::getAxonGrowthDirection() {
    return myState->axonGrowthDirection;
}

MASS_FUNCTION int NeuronPlace::getGrowthDirection() {
    return myState->growthDirection;
}

MASS_FUNCTION int NeuronPlace::getPlaceType() {
    return myState->myPlaceType;
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

MASS_FUNCTION bool NeuronPlace::isOccupied() {
    return myState->occupied;
}

MASS_FUNCTION void NeuronPlace::setOccupied(bool isOccupied) {
    myState->occupied = isOccupied;
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

MASS_FUNCTION void NeuronPlace::callMethod(int functionId, void *argument) {
	switch (functionId) {
        case SET_TIME:
            setSimulationTime((int*)argument);
        case INIT_NEURONS:
            setNeuronPlaceType((int*)argument);
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
        case FIND_GROWTH_DESTINATIONS_OUTSIDE_SOMA:
            findGrowthDestinationOutsideSoma();
            break;   
        case FIND_BRANCH_DESTINATIONS_OUTSIDE_SOMA:
            findBranchDestinationsOutsideSoma();
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
        case REMOVE_MARKED_AGENTS_FROM_PLACE:
            removeMarkedAgentsFromPlace();
            break;
        default:
			break;
    }
}