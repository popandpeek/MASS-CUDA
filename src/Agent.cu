
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
    this->state->markForDelete = false;
    this->state->childPlace = NULL;
    this->state->nChildren = 0;
    this->state->destPlace = NULL;
    this->state->destPlaceIdx = -1;
    this->state->placeAgentArrayIdx = -1;
    this->state->traveledAgentIdx = -1;
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

MASS_FUNCTION void Agent::setMyDevice(int device) {
    state->myDevice = device;
}

MASS_FUNCTION int Agent::getMyDevice() {
    return state->myDevice;
}

MASS_FUNCTION int Agent::getIndex() {
    return state->index;
}

MASS_FUNCTION void Agent::setIndex(int index) {
    state->index = index;
}

MASS_FUNCTION void Agent::setDestPlaceIndex(int idx) {
    state -> destPlaceIdx = idx;
}

MASS_FUNCTION Place* Agent::getDestPlace() {
    return state -> destPlace;
}

MASS_FUNCTION int Agent::getDestPlaceIndex() {
    return state -> destPlaceIdx;
}

MASS_FUNCTION bool Agent::isAlive() {
    return state -> isAlive;
}

MASS_FUNCTION void Agent::setAlive(bool isAlive) {
    this->state->isAlive = isAlive;
}

MASS_FUNCTION void Agent::setTraveled(bool isTraveled) {
    this->state->agentTraveled = isTraveled;
}

MASS_FUNCTION bool Agent::isTraveled() {
    return this->state->agentTraveled;
}

MASS_FUNCTION void Agent::setAccepted(bool accepted) {
    this->state->isAccepted = accepted;
}

MASS_FUNCTION bool Agent::isAccepted(){
    return state->isAccepted;
}

MASS_FUNCTION void Agent::setTraveledAgentIdx(int idx) {
    this->state->traveledAgentIdx = idx;
}

MASS_FUNCTION int Agent::getTraveledAgentIdx() {
    return state->traveledAgentIdx;
}

MASS_FUNCTION bool Agent::longDistanceMigration() {
    return state->longDistanceMigration;
}

MASS_FUNCTION void Agent::setLongDistanceMigration(bool longDistanceMigration) {
    state->longDistanceMigration = longDistanceMigration;
}

MASS_FUNCTION void Agent::setPlaceAgentArrayIdx(int idx) {
    this->state->placeAgentArrayIdx = idx;
}

MASS_FUNCTION int Agent::getPlaceAgentArrayIdx() {
    return state->placeAgentArrayIdx;
}

MASS_FUNCTION void Agent::markForTermination(bool mark) {
    this->state->markForDelete = mark;
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

MASS_FUNCTION void Agent::markAgentForTermination() {
    state -> isAlive = false;
    state-> agentTraveled = false;
    state->markForDelete = true;
    state->place = NULL;
    state -> index = -1;
}

MASS_FUNCTION bool Agent::isMarkedForTermination() {
    return state->markForDelete;
}

MASS_FUNCTION void Agent::migrateAgent(Place* destination, int destinationRelativeIdx) {
    state -> destPlace = destination;
    state -> destPlaceIdx = destination -> getDevIndex();
    setPlaceAgentArrayIdx(destinationRelativeIdx);
    destination -> addMigratingAgent(this, destinationRelativeIdx);
}

MASS_FUNCTION void Agent::migrateAgentLongDistance(Place* destination, int destinationIdx) {
    state -> longDistanceMigration = true;
    state -> destPlace = destination;
    state -> destPlaceIdx = destinationIdx;
}

MASS_FUNCTION void Agent::spawn(int numAgents, Place* place) {
    state -> nChildren += numAgents;
    state -> childPlace = place;
}

} /* namespace mass */

