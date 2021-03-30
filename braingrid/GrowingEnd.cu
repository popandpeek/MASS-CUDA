#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
# include "GrowingEnd.h"
#include "BrainGridConstants.h"


using namespace mass;
using namespace BrainGridConstants;


MASS_FUNCTION GrowingEnd::GrowingEnd(mass::AgentState *state, void *argument) :
        Agent(state, argument) {

    myState = (GrowingEndState*) state;
    myState->hasSpawned = false;
    myState->isGrowing = false;
    myState->justGrew = false;
    myState->branchCountRemaining = R_REPETITIVE_BRANCHES;
    myState->branchGrowthRemaining = K_BRANCH_GROWTH; 
}

MASS_FUNCTION GrowingEnd::~GrowingEnd() {
    // nothing to delete
}

MASS_FUNCTION void GrowingEnd::callMethod(int functionId, void *argument) { 
    switch (functionId) {
        case INIT_AXONS:
            initAxons((int*) argument);
            break;
        case INIT_DENDRITES:
            initDendrites((int*) argument);
            break;
        case SET_SPAWN_TIME:
            setSpawnTime((int*) argument);
            break;
        case SPAWN_AXONS:
            spawnAxon();
            break;
        case SPAWN_DENDRITES:
            spawnDendrite();
            break;
        case GROW_AXON_SOMA:
            growFromSoma();
            break;
        case GROW_DENDRITE_SOMA:
            growFromSoma();
            break;
        case AXON_TO_SYNAPSE:
            axonToSynapse((int*) argument);
            break;
        case GROW_AXONS_OUTSIDE_SOMA:
            growAxonsNotSoma((int*) argument);
            break;
        case BRANCH_SYNAPSES:
            branchSynapses((int*) argument);
            break;
        case BRANCH_DENDRITES:    
            branchDendrites((int*) argument);
            break;
        case GROW_SYNAPSE:
            growSynapse();
            break;    
        case GROW_DENDRITE:
            growDendrite();
            break;
        case SET_BRANCHED_SYNAPSES:
            setBranchedSynapses();
            break;
        case SET_BRANCHED_DENDRITES:
            setBranchedDendrites();
            break;
        case SET_MIGRATED_BRANCHES:
            setMigratedBranches();
            break;
        case SOMA_TRAVEL:
            somaTravel();
            break;
        case GET_SIGNAL:
            getSignal();
            break;
        case DENDRITE_SOMA_TRAVEL:
            dendriteSomaTravel();
            break;
        case SET_SOMA_SIGNAL:
            setSomaSignal();
            break;
        case UPDATE_ITER:
            updateIters();
            break;
        case default:
            break;
    }
}

MASS_FUNCTION void GrowingEnd::initAxons() {
    myState->mySoma = (NeuronPlace*) getPlace();
    myState->mySomaIndex = getPlaceIndex();
    myState->myType = AXON;

}

MASS_FUNCTION void GrowingEnd::initDendrites() {
    myState->mySoma = (NeuronPlace*) getPlace();
    myState->mySomaIndex = getPlaceIndex();
    myState->myType = DENDRITE;
}

MASS_FUNCTION void GrowingEnd::setDendriteSpawnTime() {
    NeuronPlace* myPlace = (NeuronPlace*) getPlace();
    unsigned int pos = myPlace->getCurDendritePos();
    if (pos < MAX_NEIGHBORS) {
        myState->spawnTime = myPlace->dendriteSpawnTime[pos];
    }
}

MASS_FUNCTION void GrowingEnd::setAxonSpawnTime() {
    NeuronPlace* myPlace = (NeuronPlace*) getPlace();
    myState->spawnTime = myPlace->axonSpawnTime;
}

MASS_FUNCTION void GrowingEnd::spawnAxon() {
    if (myState->spawnTime == myState->mySoma->curIter) {
        myState->hasSpawned = true;
        myState->isGrowing = true;
    }
}

MASS_FUNCTION void GrowingEnd::spawnDendrite() {
    if (myState->spawnTime == myState->mySoma->curIter) {
        myState->hasSpawned = true;
        myState->isGrowing = true;
        if (myState->mySoma->dendritesToSpawn > 0) {
            spawn(1, myState->mySoma);
            myState->mySoma->dendritesToSpawn--;
        }
    }
}

MASS_FUNCTION void GrowingEnd::growFromSoma() {
    if (myState->hasSpawned && getPlaceIndex() == getSomaIndex()) {
        myState->justGrew = true;
        NeuronPlace* myPlace = (NeuronPlace*) getPlace();
        if (myPlace->getMigrationDest() != NULL) {
            migrateAgent(myPlace->getMigrationDest(), myPlace->getMigrationDestRelIdx());
        }
    }
}

MASS_FUNCTION void GrowingEnd::setNewlyGrownSynapse() {
    if (myState->justGrew) {
        ((NeuronPlace*)myState->place)->travelingSynapse = this;
    }
}

MASS_FUNCTION void GrowingEnd::setNewlyGrownDendrites() {
    if (myState->justGrew) {
        ((NeuronPlace*)myState->place)->travelingDendrite = this;
    }
}

MASS_FUNCTION void GrowingEnd::axonToSynapse(int* randNums) {
    if (myState->isGrowing && myState->mySoma != getIndex()) {
        
        int randNum = (randNums[getIndex()] % 100;
        if (randNum >= AXON_GROWING_MODE) {
            if (myState->place->travelingSynapse == NULL || 
                    myState->mySomaIndex < myState->place->travelingSynapse->mySomaIndex) {
                myState->place->travelingSynapse = this;
            } 

            myState->myType = SYNAPSE;
            branchCountRemaining = R_REPETITIVE_BRANCHES;
            branchGrowthRemaining = K_BRANCH_GROWTH;
        }
    }
}

MASS_FUNCTION void GrowingEnd::growAxonsNotSoma(int* randNums) {
    if (myState->place->index != myState->mySomaIndex && myState->myState->myType == AXON && myState->isGrowing) {
        randNum = randNums[getIndex()] % 100;
        if (randNum >= AXON_GROWING_MODE)
            myState->isGrowing = false;
            return;
        }

        int* potDests[R_REPETITIVE_BRANCHES];
        potDests[0] = myState->growthDirection;
        potDests[1] = (myState->growthDirection - 1) % MAX_NEIGHBORS;
        potDests[2] = (myState->growthDirection + 1) % MAX_NEIGHBORS;
        for (int i = 0; i < R_REPETITIVE_BRANCHES; ++i) {
            myState->growthDirection = potDests[i];
            NeuronPlace* neighborPlace = (NeuronPlace*) myState->place->neighbors[myState->growthDirection];
            if (neighborPlace != NULL && !(neighborPlace->occupied)) {
                myState->justGrew = true;
                migrateAgent(myState->place->neighbors[myState->growthDirection])
                return;
            }
        }
        
        terminateAgent();
    }
}

MASS_FUNCTION void GrowingEnd::branchSynapses(int* randNums) {
    if (myState->mytype == SYNAPSE && myState->isGrowing) {
        int myLeftNeighborDir = (myState->growthDirection - 1) % MAX_NEIGHBORS;
        int myRightNeighborDir = (myState->growthDirection + 1) % MAX_NEIGHBORS;
        int numToSpawn = 0;
        NeuronPlace* myLeftNeighbor = myState->myPlace->neighbors[myLeftNeighborDir];
        NeuronPlace* myRightNeighbor = myState->myPlace->neighbors[myRightNeighborDir];
        if (myLeftNeighbor != NULL && randNums[i * 2] % 100 < BRANCH_POSSIBILITY && 
                myLeftNeighbor->growthInSomas[(myLeftNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] == NULL) {
            myLeftNeighbor->growthInSomas[(myLeftNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->mySoma;
            myLeftNeighbor->growthInSomasType[(myLeftNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->myType;
            numToSpawn += 1;
        } 
        if (myRightNeighbor != NULL && randNums[i * 2 + 1] % 100 < BRANCH_POSSIBILITY && 
                myRightNeighbor->growthInSomas[myRightNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] == NULL {
            myRightNeighbor->growthInSomas[(myRightNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->mySoma
            myRightNeighbor->growthInSomasType[(myRightNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->myType
            numToSpawn += 1;
        }
        if (numToSpawn > 0) {
            spawn(numToSpawn, myState->place);
        }
    }    
}

MASS_FUNCTION void GrowingEnd::branchDendrites(int* randNums) {
    if (myState->mytype == DENDRITE && myState->isGrowing) {
        int myLeftNeighborDir = (myState->growthDirection - 1) % MAX_NEIGHBORS;
        int myRightNeighborDir = (myState->growthDirection + 1) % MAX_NEIGHBORS;
        int numToSpawn = 0;
        NeuronPlace* myLeftNeighbor = myState->myPlace->neighbors[myLeftNeighborDir];
        NeuronPlace* myRightNeighbor = myState->myPlace->neighbors[myRightNeighborDir];
        if (myLeftNeighbor != NULL && randNums[i * 2] % 100 < BRANCH_POSSIBILITY && 
                myLeftNeighbor->growthInSomas[(myLeftNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] == NULL) {
            myLeftNeighbor->growthInSomas[(myLeftNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->mySoma;
            myLeftNeighbor->growthInSomasType[(myLeftNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->myType;
            numToSpawn += 1;
        } 
        if (myRightNeighbor != NULL && randNums[i * 2 + 1] % 100 < BRANCH_POSSIBILITY && 
                myRightNeighbor->growthInSomas[myRightNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] == NULL {
            myRightNeighbor->growthInSomas[(myRightNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->mySoma;
            myRightNeighbor->growthInSomasType[(myRightNeighborDir + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS] = myState->myType;
            numToSpawn += 1;
        }
        if (numToSpawn > 0) {
            spawn(numToSpawn, myState->place);
        }
    }
}

MASS_FUNCTION void GrowingEnd::growSynapse() {
    if (myState->isGrowing) {
        NeuronPlace* growPlace = (NeuronPlace*) myState->place->neighbors[myState->growthDirection];
        if (growPlace != NULL && growPlace->travelingSynapse == NULL) {
            myState->justGrew = true;
            myState->branchGrowthRemaining--;
            // TODO: Is the below the correct numbering scheme?
            migrateAgent(growPlace, (myState->growthDirection + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS);
        }
    }
}

MASS_FUNCTION void GrowingEnd::growDendrite() {
    if (myState->isGrowing) {
        NeuronPlace* growPlace = (NeuronPlace*) myState->place->neighbors[myState->growthDirection];
        if (growPlace != NULL && growPlace->travelingSynapse == NULL) {
            myState->justGrew = true;
            myState->branchGrowthRemaining--;
            // TODO: Is the below the correct numbering scheme?
            migrateAgent(growPlace, (myState->growthDirection + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS);
        }
    }
}

MASS_FUNCTION void GrowingEnd::setBranchedSynapses() {
    if (!myState->hasSpawned && myState->mySoma == NULL) {
        myState->myType = SYNAPSE;
        myState->hasSpawned = true;
        myState->isGrowing = true;
    }
}

MASS_FUNCTION void GrowingEnd::setBranchedDendrites() {
    if (!myState->hasSpawned && myState->mySoma == NULL) {
        myState->myType = DENDRITE;
        myState->hasSpawned = true;
        myState->isGrowing = true;
    }
}

MASS_FUNCTION void GrowingEnd::growBranches() {
    if (myState->isGrowing && !(myState->justGrew)) {
        NeuronPlace* growPlace = (NeuronPlace*) myState->place->neighbors[myState->growthDirection];
        if (growPlace != NULL) {
            myState->justGrew = true;
            myState->branchGrowthRemaining--;
            // TODO: Is the below the correct numbering scheme?
            migrateAgent(growPlace, (myState->growthDirection + MAX_NEIGHBORS / 2) % MAX_NEIGHBORS);
        }
    }
}

MASS_FUNCTION void GrowingEnd::checkGrowingEndGrowth() {
    if (myState->isAlive && myState->isGrowing) {
        if (myState->branchGrowthRemaining == 0) {
            myState->isGrowing = false;
        }
    }
}

MASS_FUNCTION void somaTravel() {
    // TODO: Is connected?
    if (myState->isAlive && !(myState->isGrowing) && (myState->myType == SYNAPSE) {
        // TODO: Need to provide below method signature in Agent class
        migrateAgent(myState->mySoma);
    }
}

MASS_FUNCTION void GrowingEnd::getSignal() {
    if (myState->myType == SYNAPSE && myState->isAlive && !myState->isGrowing && 
                myState->getPlaceIndex() == myState->mySomaIndex) {
        myState->signal = myState->place->outputSignal;
    }
}

MASS_FUNCTION void GrowingEnd::dendriteSomaTravel() {
    if (myState->isAlive && !(myState->isGrowing) && (myState->myType == SYNAPSE) {
        migrateAgent(myState->myConnectPlace);
    }
}