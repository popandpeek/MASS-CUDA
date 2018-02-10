
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
}

MASS_FUNCTION AgentState* Agent::getState() {
    return state;
}

MASS_FUNCTION Place* Agent::getPlace() {
    return state->place;
}

MASS_FUNCTION void Agent::setPlace(Place* place) {
    state->place = place;
}

MASS_FUNCTION int Agent::getPlaceIndex() {
    if (state->place != NULL) {
        return state->place->getIndex();
    } else {
        printf("Warning: Agent[%d] place is NULL\n", getIndex());
        return -1;
    }
    
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
    state -> isAlive = false;
    state->place->removeAgent(this);
}

MASS_FUNCTION void Agent::migrateAgent(Place* destination, int destinationRelativeIdx) {
    state -> destPlace = destination;
    destination -> addMigratingAgent(this, destinationRelativeIdx);
}

} /* namespace mass */

