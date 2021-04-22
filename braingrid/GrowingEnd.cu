#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "GrowingEnd.h"
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

MASS_FUNCTION bool GrowingEnd::isGrowing() {
    return myState->isGrowing;
}

MASS_FUNCTION void GrowingEnd::setGrowing(bool grow) {
    myState->isGrowing = grow;
}

MASS_FUNCTION int GrowingEnd::getType() {
    return myState->myType;
}

MASS_FUNCTION int GrowingEnd::getSomaIndex() {
    return myState->mySomaIndex;
}

MASS_FUNCTION NeuronPlace* GrowingEnd::getSoma() {
    return myState->mySoma;
}

MASS_FUNCTION void GrowingEnd::setSomaIndex(int idx) {
    myState->mySomaIndex = idx;
}

MASS_FUNCTION void GrowingEnd::setSoma(NeuronPlace* nSoma) {
    myState->mySoma = nSoma;
}

MASS_FUNCTION int GrowingEnd::getConnectPlaceIndex() {
    return myState->myConnectPlaceIndex;
}

MASS_FUNCTION NeuronPlace* GrowingEnd::getConnectPlace() {
    return myState->myConnectPlace;
}

MASS_FUNCTION void GrowingEnd::setConnectPlaceIndex(int idx) {
    myState->myConnectPlaceIndex = idx;
}

MASS_FUNCTION void GrowingEnd::setConnectPlace(NeuronPlace* connect) {
    myState->myConnectPlace = connect;
}

MASS_FUNCTION double GrowingEnd::getSignal() {
    return myState->signal;
}

MASS_FUNCTION void GrowingEnd::setSignal(double nSignal) {
    myState->signal = nSignal;
}

MASS_FUNCTION bool GrowingEnd::justGrew() {
    return myState->justGrew;
}

MASS_FUNCTION void GrowingEnd::setGrowthDirection(int direction) {
    myState->growthDirection = (BrainGridConstants::Direction)direction;
}

MASS_FUNCTION void GrowingEnd::callMethod(int functionId, void *argument) { 
    switch (functionId) {
        case INIT_AXONS:
            initAxons();
            break;
        case INIT_DENDRITES:
            initDendrites();
            break;
        case SET_SPAWN_TIME:
            setDendriteSpawnTime();
            setAxonSpawnTime();
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
            growSynapsesNotSoma((int*) argument);
            break;    
        case GROW_DENDRITE:
            growDendritesNotSoma((int*) argument);
            break;
        case GROW_BRANCHES:
            growBranches();
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
        case UPDATE_ITERS:
            updateIters();
            break;
        default:
            break;
    }
}

MASS_FUNCTION void GrowingEnd::initAxons() {
    myState->mySoma = (NeuronPlace*) getPlace();
    myState->mySomaIndex = getPlaceIndex();
    myState->growthDirection = (BrainGridConstants::Direction)(myState->mySoma->getGrowthDirection());
    myState->myType = AXON;

}

MASS_FUNCTION void GrowingEnd::initDendrites() {
    myState->mySoma = (NeuronPlace*) getPlace();
    myState->mySomaIndex = getPlaceIndex();
    myState->growthDirection = (BrainGridConstants::Direction)(myState->mySoma->getGrowthDirection());
    myState->myType = DENDRITE;
}

MASS_FUNCTION void GrowingEnd::setDendriteSpawnTime() {
    NeuronPlace* myPlace = (NeuronPlace*) getPlace();
    myState->spawnTime = myPlace->getDendriteSpawnTime();
}

MASS_FUNCTION void GrowingEnd::setAxonSpawnTime() {
    NeuronPlace* myPlace = (NeuronPlace*) getPlace();
    myState->spawnTime = myPlace->getAxonSpawnTime();
}

MASS_FUNCTION void GrowingEnd::spawnAxon() {
    if (!(myState->hasSpawned) && myState->spawnTime == myState->mySoma->getCurIter()) {
        myState->hasSpawned = true;
        myState->isGrowing = true;
    }
}

MASS_FUNCTION void GrowingEnd::spawnDendrite() {
    if (!(myState->hasSpawned) && myState->spawnTime == myState->mySoma->getCurIter()) {
        myState->hasSpawned = true;
        myState->isGrowing = true;
        if (myState->mySoma->getDendritesToSpawn() > 0) {
            spawn(1, myState->mySoma);
            myState->mySoma->reduceDendritesToSpawn(1);
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

// TODO: Refactor: To use travelingSynapse in earlier calls? 
MASS_FUNCTION void GrowingEnd::axonToSynapse(int* randNums) {
    if (myState->isGrowing && myState->myType == AXON && !(justGrew()) && myState->mySomaIndex != getIndex()) {
        int randNum = (randNums[getIndex()]) % 100;
        if (randNum >= AXON_GROWING_MODE) {
            myState->myType = SYNAPSE;
            myState->branchCountRemaining = R_REPETITIVE_BRANCHES;
            myState->branchGrowthRemaining = K_BRANCH_GROWTH;
            if (((NeuronPlace*)myState->place)->getTravelingSynapse() == NULL) {
                ((NeuronPlace*)myState->place)->setTravelingSynapse(this);
            } 
            else {
                terminateAgent();
            }
        }
    }
}

// TODO: Refactor to use place growthDirection?
MASS_FUNCTION void GrowingEnd::growAxonsNotSoma(int* randNums) {
    if (getIndex() != myState->mySomaIndex && myState->myType == AXON && 
            myState->isGrowing && !(justGrew())) {
        int randNum = randNums[getIndex()] % 100;
        if (randNum >= AXON_GROWING_MODE) {
            terminateAgent();
            return;
        }

        int potDests[R_REPETITIVE_BRANCHES];
        potDests[0] = myState->growthDirection;
        potDests[1] = (myState->growthDirection - 1) % MAX_NEIGHBORS;
        potDests[2] = (myState->growthDirection + 1) % MAX_NEIGHBORS;
        for (int i = 0; i < R_REPETITIVE_BRANCHES; ++i) {
            myState->growthDirection = (Direction)potDests[i];
            NeuronPlace* neighborPlace = ((NeuronPlace*)(myState->place->state->neighbors[myState->growthDirection]));
            if (neighborPlace != NULL && !(neighborPlace->isOccupied())) {
                myState->justGrew = true;
                migrateAgent(myState->place->state->neighbors[myState->growthDirection], myState->growthDirection);
                return;
            }
        }   
        
        terminateAgent();
    }
}

MASS_FUNCTION void GrowingEnd::branchSynapses(int* randNums) {
    if (myState->myType == SYNAPSE && isGrowing() && !(justGrew())) {
        int randNum = randNums[getIndex()];
        if (randNum % 100 < BRANCH_POSSIBILITY && myState->branchCountRemaining > 0) {
            myState->branchCountRemaining--;
            ((NeuronPlace*)myState->place)->setBranchedSynapseSoma(myState->mySoma);
            ((NeuronPlace*)myState->place)->setBranchedSynapseSomaIdx(myState->mySomaIndex);
            spawn(1, myState->place);
        }
    }
}

MASS_FUNCTION void GrowingEnd::branchDendrites(int* randNums) {
    if (myState->myType == DENDRITE && isGrowing() && !(justGrew())) {
        int randNum = randNums[getIndex()];
        if (randNum % 100 < BRANCH_POSSIBILITY && myState->branchCountRemaining > 0) {
            myState->branchCountRemaining--;
            ((NeuronPlace*)myState->place)->setBranchedDendriteSoma(myState->mySoma);
            ((NeuronPlace*)myState->place)->setBranchedDendriteSomaIdx(myState->mySomaIndex);
            spawn(1, myState->place);
        }
    }
}

// TODO: Where are we setting SOMA? Can we combine?
MASS_FUNCTION void GrowingEnd::setBranchedSynapses() {
    if (!myState->hasSpawned && myState->mySoma == NULL) {
        myState->myType = SYNAPSE;
        myState->hasSpawned = true;
        myState->isGrowing = true;
        myState->mySoma = ((NeuronPlace*)myState->place)->getBranchedSynapseSoma();
        myState->mySomaIndex = ((NeuronPlace*)myState->place)->getBranchedSynapseSomaIdx();
    }
}

// TODO: Where are we setting SOMA Can we combine?
MASS_FUNCTION void GrowingEnd::setBranchedDendrites() {
    if (!myState->hasSpawned && myState->mySoma == NULL) {
        myState->myType = DENDRITE;
        myState->hasSpawned = true;
        myState->isGrowing = true;
        myState->mySoma = ((NeuronPlace*)myState->place)->getBranchedDendriteSoma();
        myState->mySomaIndex = ((NeuronPlace*)myState->place)->getBranchedDendriteSomaIdx();
    }
}

// TODO: Refacctor to get rid of check branchGrowthRemaining?
MASS_FUNCTION void GrowingEnd::growSynapsesNotSoma(int* randNums) {
    if (myState->myType == SYNAPSE && myState->isGrowing && !(justGrew())) { 
        NeuronPlace* growPlace = ((NeuronPlace*)myState->place)->getMigrationDest();
        if (growPlace != NULL && myState->branchGrowthRemaining > 0 && (randNums[getIndex()] % 100) > STOP_BRANCH_GROWTH) {
            myState->branchGrowthRemaining--;
            migrateAgent(growPlace, ((NeuronPlace*)myState->place)->getMigrationDestRelIdx());
        }
        else {
            terminateAgent();
        }
    }
}

// TODO: Refacctor to get rid of check branchGrowthRemaining?
MASS_FUNCTION void GrowingEnd::growDendritesNotSoma(int* randNums) {
    if (myState->isGrowing && !(justGrew())) { 
        NeuronPlace* growPlace = ((NeuronPlace*)myState->place)->getMigrationDest();
        if (growPlace != NULL && myState->branchGrowthRemaining > 0 && (randNums[getIndex()] % 100) > STOP_BRANCH_GROWTH) {
            myState->branchGrowthRemaining--;
            migrateAgent(growPlace, ((NeuronPlace*)myState->place)->getMigrationDestRelIdx());
        }
        else {
            terminateAgent();
        }
    }
}

MASS_FUNCTION void GrowingEnd::growBranches() {
    if (myState->isGrowing && !(myState->justGrew)) {
        int dir1 = (((NeuronPlace*)myState->place)->getGrowthDirection() - 1) % MAX_NEIGHBORS;
        int dir2 = (((NeuronPlace*)myState->place)->getGrowthDirection() + 1) % MAX_NEIGHBORS;
        NeuronPlace* growPlace1 = (NeuronPlace*) myState->place->state->neighbors[dir1];
        NeuronPlace* growPlace2 = (NeuronPlace*) myState->place->state->neighbors[dir2];
        if (growPlace1 != NULL && !(growPlace1->isOccupied())) {
            myState->justGrew = true;
            myState->branchGrowthRemaining--;
            migrateAgent(growPlace1, dir1);
        } else if (growPlace2 != NULL && !(growPlace2->isOccupied())) {
            myState->justGrew = true;
            myState->branchGrowthRemaining--;
            migrateAgent(growPlace2, dir2);
        } else {
            terminateAgent();
        }
    }
}

MASS_FUNCTION void GrowingEnd::somaTravel() {
    // TODO: Is connected?
    if (myState->isAlive && !(myState->isGrowing) && (myState->myType == SYNAPSE)) {
        // TODO: Need to provide below method signature in Agent class
        migrateAgentLongDistance(myState->mySoma, myState->mySomaIndex);
    }
}

MASS_FUNCTION void GrowingEnd::neuronGetSignal() {
    if (myState->myType == SYNAPSE && myState->isAlive && !(isGrowing()) && 
                getPlaceIndex() == myState->mySomaIndex) {
        myState->signal = ((NeuronPlace*)myState->place)->getOutputSignal();
    }
}

MASS_FUNCTION void GrowingEnd::dendriteSomaTravel() {
    if (myState->isAlive && !(myState->isGrowing) && (myState->myType == SYNAPSE)) {
        migrateAgentLongDistance(myState->myConnectPlace, myState->myConnectPlaceIndex);
    }
}

MASS_FUNCTION void GrowingEnd::setSomaSignal() {
    if (myState->myType == SYNAPSE && myState->isAlive && !myState->isGrowing && 
            getPlaceIndex() == myState->mySomaIndex) {
    }
}

MASS_FUNCTION void GrowingEnd::updateIters() {
    if (myState->isGrowing) {
        myState->justGrew = false;
    }
}

