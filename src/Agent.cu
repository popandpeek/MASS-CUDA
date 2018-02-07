
#include "Agent.h"  
#include "AgentState.h"
#include <stdio.h>

namespace mass {

/**
 *  A contiguous space of arguments is passed
 *  to the constructor.
 */MASS_FUNCTION Agent::Agent(AgentState *state, void *args) {
    this->state = state;
    this->state->index = 0;

    this->state->destRelativeIdx = -1;
}

MASS_FUNCTION AgentState* Agent::getState() {
    return state;
}

MASS_FUNCTION Place* Agent::getPlace() {
    return state->place;
}

MASS_FUNCTION void Agent::setPlace(Place* place) {
    state->place = place;
    state->placeIndex = place->getIndex();
}

MASS_FUNCTION int Agent::getPlaceIndex() {
    return state->placeIndex;
}    

MASS_FUNCTION int Agent::getIndex() {
    return state->index;
}

MASS_FUNCTION void Agent::setIndex(int index) {
    state->index = index;
}

MASS_FUNCTION void Agent::setSize(int qty) {
    state->size = qty;
}

MASS_FUNCTION int Agent::getSize() {
    return state->size;
}

MASS_FUNCTION bool Agent::isAlive() {
    return state -> isAlive;
}

MASS_FUNCTION void Agent::setAlive() {
    this->state->isAlive = true;
}

MASS_FUNCTION void Agent::terminateAgent() {
    // printf("__________Terminating agent %d\n", getIndex());
    state -> isAlive = false;
}

MASS_FUNCTION void Agent::migrateAgent(Place* destination, int destinationRelativeIdx) {
    // printf("__________Attempting to migrate agent %d from %d to %d\n", getIndex(), getPlaceIndex(), destination->getIndex());
    state -> destPlace = destination;
    state -> destRelativeIdx = destinationRelativeIdx;

    destination -> addMigratingAgent(this, destinationRelativeIdx);
}

} /* namespace mass */

