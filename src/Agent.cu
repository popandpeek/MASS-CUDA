
#include "Agent.h"  
#include "AgentState.h"
#include <stdio.h>

namespace mass {

/**
*  A contiguous space of arguments is passed
*  to the constructor.
*/
MASS_FUNCTION Agent::Agent(AgentState *state, void *args) {
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
    state->placeIndex = place->getIndex();
    state->placeDevIndex = place->getDevIndex();
}

MASS_FUNCTION int Agent::getPlaceIndex() {
    return state->placeIndex;
}    

MASS_FUNCTION int Agent::getPlaceDevIndex() {
    return state->placeDevIndex;
}

MASS_FUNCTION int Agent::getIndex() {
    return state->index;
}

MASS_FUNCTION void Agent::setIndex(int index) {
    state->index = index;
}

MASS_FUNCTION bool Agent::isAlive() {
    return state -> isAlive;
}

MASS_FUNCTION void Agent::setAlive() {
    this->state->isAlive = true;
}

MASS_FUNCTION void Agent::setTraveled(bool isTraveled) {
    this->state->agentTraveled = isTraveled;
}

MASS_FUNCTION bool Agent::isTraveled() {
    return this->state->agentTraveled;
}

MASS_FUNCTION void Agent::terminateAgent() {
    state -> isAlive = false;
    state-> agentTraveled = false;
    state->place->removeAgent(this);
    state -> index = -1;
}

MASS_FUNCTION void Agent::terminateGhostAgent() {
    state -> isAlive = false;
    state-> agentTraveled = false;
    state->place = NULL;
    state -> index = -1;
}


MASS_FUNCTION void Agent::migrateAgent(Place* destination, int destinationRelativeIdx) {
    state -> destPlace = destination;
    destination -> addMigratingAgent(this, destinationRelativeIdx);
}

MASS_FUNCTION void Agent::spawn(int numAgents, Place* place) {
    
    state -> nChildren += numAgents;
    state -> childPlace = place;
}

} /* namespace mass */

