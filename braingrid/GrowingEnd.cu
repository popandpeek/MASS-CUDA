#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "GrowingEnd.h"
#include "BrainGridConstants.h"


using namespace mass;

MASS_FUNCTION GrowingEnd::GrowingEnd(mass::AgentState *state, void *argument) :
        Agent(state, argument) {

    myState = (GrowingEndState*) state;
    myState->myAgentType = 0;
    myState->hasSpawned = false;
    myState->isGrowing = false;
    myState->justGrew = false;
    myState->branchCountRemaining = BrainGridConstants::R_REPETITIVE_BRANCHES;
    myState->branchGrowthRemaining = BrainGridConstants::K_BRANCH_GROWTH; 
    myState->mySoma = NULL;
    myState->mySomaIndex = 0;
    myState->myConnectPlace = NULL;
    myState->curIter = 0;
    myState->spawnTime = -1;
}

MASS_FUNCTION GrowingEnd::~GrowingEnd() {
    // nothing to delete
}

MASS_FUNCTION void GrowingEnd::resetGrowingEnd() {
    myState->longDistanceMigration = false;
    myState->hasSpawned = false;
    myState->isGrowing = false;
    myState->justGrew = false;
    myState->branchCountRemaining = BrainGridConstants::R_REPETITIVE_BRANCHES;
    myState->branchGrowthRemaining = BrainGridConstants::K_BRANCH_GROWTH; 
    myState->mySoma = NULL;
    myState->myConnectPlace = NULL;
    myState->curIter = 0;
    myState->spawnTime = -1;
}

MASS_FUNCTION GrowingEndState* GrowingEnd::getState() {
    return myState;
}

MASS_FUNCTION bool GrowingEnd::isGrowing() {
    return myState->isGrowing;
}

MASS_FUNCTION void GrowingEnd::setGrowing(bool grow) {
    myState->isGrowing = grow;
}

MASS_FUNCTION int GrowingEnd::getAgentType() {
    return myState->myAgentType;
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

MASS_FUNCTION int GrowingEnd::getGrowthDirection() {
    return myState->growthDirection;
}

MASS_FUNCTION void GrowingEnd::setGrowthDirection(int direction) {
    myState->growthDirection = direction;
}

MASS_FUNCTION void GrowingEnd::callMethod(int functionId, void *argument) { 
    switch (functionId) {
        case INIT_AXONS:
            initAxons();
            break;
        case INIT_DENDRITES:
            initDendrites();
            break;
        case SET_AXON_SPAWN_TIME:
            setAxonSpawnTime();
            break;
        case SET_DENDRITE_SPAWN_TIME:
            setDendriteSpawnTime();
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
        case SET_SYNAPSES:
            setSynapses();
            break;
        case SET_DENDRITES:
            setDendrites();
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
    myState->growthDirection = myState->mySoma->getAxonGrowthDirection();
    myState->myAgentType = BrainGridConstants::AXON;
    setAxonSpawnTime();

}

MASS_FUNCTION void GrowingEnd::initDendrites() {
    if (!(myState->hasSpawned)) {
        myState->mySoma = (NeuronPlace*)myState->place;
        myState->mySomaIndex = getPlaceIndex();
        myState->curIter = myState->mySoma->getCurIter();
        myState->growthDirection = myState->mySoma->getGrowthDirection();
        myState->myAgentType = BrainGridConstants::DENDRITE;
        setDendriteSpawnTime();
    }
}

MASS_FUNCTION void GrowingEnd::setDendriteSpawnTime() {
    if (!(myState->hasSpawned)) {
        myState->spawnTime = myState->mySoma->getDendriteSpawnTime();
    }
}

MASS_FUNCTION void GrowingEnd::setAxonSpawnTime() {
    if (!(myState->hasSpawned)) {
        myState->spawnTime = myState->mySoma->getAxonSpawnTime();
    }
}

MASS_FUNCTION void GrowingEnd::spawnAxon() {
    if (!(myState->hasSpawned) && (myState->spawnTime == myState->curIter)) {
        myState->hasSpawned = true;
        myState->isGrowing = true;
    }
}

MASS_FUNCTION void GrowingEnd::spawnDendrite() {
    if (!(myState->hasSpawned) && (myState->spawnTime <= myState->curIter)) {
        myState->hasSpawned = true;
        myState->isGrowing = true;
        if (myState->mySoma->getDendritesToSpawn() > 0) {
            spawn(1, myState->place);
            myState->mySoma->reduceDendritesToSpawn(1);
        }
    }
}

MASS_FUNCTION void GrowingEnd::growFromSoma() {
    if (myState->hasSpawned && getPlaceIndex() == getSomaIndex()) {
        NeuronPlace* myPlace = (NeuronPlace*) getPlace();
        if (myPlace->getMigrationDest() != NULL) {
            myState->justGrew = true;
            migrateAgent(myPlace->getMigrationDest(), myPlace->getMigrationDestRelIdx());
        }
    }
}

MASS_FUNCTION void GrowingEnd::axonToSynapse(int* randNums) {
    if (myState->isGrowing && myState->myAgentType == BrainGridConstants::AXON && 
            myState->mySomaIndex != getPlaceIndex()) {
        int randNum = (randNums[getIndex()]) % 100;
        if (randNum >= BrainGridConstants::AXON_GROWING_MODE) {
            myState->myAgentType = BrainGridConstants::SYNAPSE;
            myState->branchCountRemaining = BrainGridConstants::R_REPETITIVE_BRANCHES;
            myState->branchGrowthRemaining = BrainGridConstants::K_BRANCH_GROWTH;
        }
    }
}

MASS_FUNCTION void GrowingEnd::growAxonsNotSoma(int* randNums) {
    if (getIndex() != myState->mySomaIndex && myState->myAgentType == BrainGridConstants::AXON && 
            myState->isGrowing && !(justGrew())) {
        int randNum = randNums[getIndex()] % 100;
        if (randNum >= BrainGridConstants::AXON_GROWING_MODE) {
            resetGrowingEnd();
            markAgentForTermination();
            return;
        }

        if (((NeuronPlace*)myState->place)->getMigrationDest() != NULL) {
            migrateAgent(((NeuronPlace*)myState->place)->getMigrationDest(), 
                ((NeuronPlace*)myState->place)->getMigrationDestRelIdx());
        } else {
            resetGrowingEnd();
            markAgentForTermination();
        }
    }
}

MASS_FUNCTION void GrowingEnd::branchSynapses(int* randNums) {
    if (myState->myAgentType == BrainGridConstants::SYNAPSE && isGrowing()) {
        int randNum = randNums[getIndex()];
        if (randNum % 100 < BrainGridConstants::BRANCH_POSSIBILITY && myState->branchCountRemaining > 0) {
            myState->branchCountRemaining--;
            ((NeuronPlace*)myState->place)->setBranchedSynapseSoma(myState->mySoma);
            ((NeuronPlace*)myState->place)->setBranchedSynapseSomaIdx(myState->mySomaIndex);
            spawn(1, myState->place);
        }
    }
}

MASS_FUNCTION void GrowingEnd::branchDendrites(int* randNums) {
    if (myState->myAgentType == BrainGridConstants::DENDRITE && isGrowing()) {
        int randNum = randNums[getIndex()];
        if (randNum % 100 < BrainGridConstants::BRANCH_POSSIBILITY && myState->branchCountRemaining > 0) {
            myState->branchCountRemaining--;
            ((NeuronPlace*)myState->place)->setBranchedDendriteSoma(myState->mySoma);
            ((NeuronPlace*)myState->place)->setBranchedDendriteSomaIdx(myState->mySomaIndex);
            spawn(1, myState->place);
        }
    }
}

// TODO: Where are we setting SOMA? Can we combine?
MASS_FUNCTION void GrowingEnd::setSynapses() {
    if (!myState->hasSpawned && myState->mySoma == NULL) {
        myState->myAgentType = BrainGridConstants::SYNAPSE;
        myState->hasSpawned = true;
        myState->isGrowing = true;
        myState->mySoma = ((NeuronPlace*)myState->place)->getBranchedSynapseSoma();
        myState->mySomaIndex = ((NeuronPlace*)myState->place)->getBranchedSynapseSomaIdx();
        myState->growthDirection = ((NeuronPlace*)myState->place)->getBranchMigrationDestRelIdx();
    }
}

// TODO: Where are we setting SOMA Can we combine?
MASS_FUNCTION void GrowingEnd::setDendrites() {
    if (!myState->hasSpawned && myState->mySoma == NULL) {
        myState->myAgentType = BrainGridConstants::DENDRITE;
        myState->hasSpawned = true;
        myState->isGrowing = true;
        myState->mySoma = ((NeuronPlace*)myState->place)->getBranchedDendriteSoma();
        myState->mySomaIndex = ((NeuronPlace*)myState->place)->getBranchedDendriteSomaIdx();
        myState->growthDirection = ((NeuronPlace*)myState->place)->getBranchMigrationDestRelIdx();
    }
}

// TODO: Refacctor to get rid of check branchGrowthRemaining?
MASS_FUNCTION void GrowingEnd::growSynapsesNotSoma(int* randNums) {
    if (getAgentType() == BrainGridConstants::SYNAPSE && myState->isGrowing && justGrew()) { 
        NeuronPlace* growPlace = ((NeuronPlace*)myState->place)->getMigrationDest();
        if (growPlace != NULL && myState->branchGrowthRemaining > 0 && 
                (randNums[getIndex()] % 100) > BrainGridConstants::STOP_BRANCH_GROWTH) {
            myState->branchGrowthRemaining--;
            migrateAgent(growPlace, ((NeuronPlace*)myState->place)->getMigrationDestRelIdx());
        }
        else {
            resetGrowingEnd();
            markAgentForTermination();
        }
    }
}

// TODO: Refacctor to get rid of check branchGrowthRemaining?
MASS_FUNCTION void GrowingEnd::growDendritesNotSoma(int* randNums) {
    if (myState->isGrowing && !(justGrew())) { 
        NeuronPlace* growPlace = ((NeuronPlace*)myState->place)->getMigrationDest();
        if (growPlace != NULL && myState->branchGrowthRemaining > 0 && 
                (randNums[getIndex()] % 100) > BrainGridConstants::STOP_BRANCH_GROWTH) {
            myState->branchGrowthRemaining--;
            migrateAgent(growPlace, ((NeuronPlace*)myState->place)->getMigrationDestRelIdx());
        }
        else {
            resetGrowingEnd();
            markAgentForTermination();
        }
    }
}

MASS_FUNCTION void GrowingEnd::growBranches() {
    if (myState->isGrowing && !(myState->justGrew) && 
            (getAgentType() == BrainGridConstants::DENDRITE || getAgentType() == BrainGridConstants::SYNAPSE)) {
        NeuronPlace* branchGrowthPlace = ((NeuronPlace*)myState->place)->getBranchMigrationDest();
        if (branchGrowthPlace != NULL) {
            myState->justGrew = true;
            myState->branchGrowthRemaining--;
            migrateAgent(branchGrowthPlace, ((NeuronPlace*)myState->place)->getBranchMigrationDestRelIdx());
        } else {
            resetGrowingEnd();
            markAgentForTermination();
        }
    }
}

MASS_FUNCTION void GrowingEnd::somaTravel() {
    // TODO: Is connected?
    if (myState->isAlive && !(myState->isGrowing) && (myState->myAgentType == BrainGridConstants::SYNAPSE)) {
        migrateAgentLongDistance(myState->mySoma, myState->mySomaIndex);
    }
}

MASS_FUNCTION void GrowingEnd::neuronGetSignal() {
    if (myState->myAgentType == BrainGridConstants::SYNAPSE && myState->isAlive && !(isGrowing()) && 
                getPlaceIndex() == myState->mySomaIndex) {
        myState->signal = ((NeuronPlace*)myState->place)->getOutputSignal();
    }
}

MASS_FUNCTION void GrowingEnd::dendriteSomaTravel() {
    if (myState->isAlive && !(myState->isGrowing) && (myState->myAgentType == BrainGridConstants::SYNAPSE)) {
        migrateAgentLongDistance(myState->myConnectPlace, myState->myConnectPlaceIndex);
    }
}

// TODO: Not sure this is this needed?
MASS_FUNCTION void GrowingEnd::setSomaSignal() {
    if (myState->myAgentType == BrainGridConstants::SYNAPSE && myState->isAlive && !myState->isGrowing && 
            getPlaceIndex() == myState->mySomaIndex) {
        // TODO: Set signal
    }
}

MASS_FUNCTION void GrowingEnd::updateIters() {
    if (myState->isGrowing) {
        myState->justGrew = false;
    }
    myState->curIter++;
}

