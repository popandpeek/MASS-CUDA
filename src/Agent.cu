
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
    //this->state->message_size = 0;
    // memset(this->state->neighbors, 0, MAX_NEIGHBORS);
    // memset(this->state->inMessages, 0, MAX_NEIGHBORS);
    // memset(this->state->size, 0, MAX_DIMS);
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

MASS_FUNCTION int Agent::getIndex() {
    return state->index;
}

MASS_FUNCTION void Agent::setIndex(int index) {
    state->index = index;
}

MASS_FUNCTION void Agent::setSize(int qty) {
    state->size = qty;
}

} /* namespace mass */

