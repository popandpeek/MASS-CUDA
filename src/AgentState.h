
#ifndef AGENTSTATE_H_
#define AGENTSTATE_H_

#include "Place.h"

namespace mass {

class AgentState {
    friend class Agent;

public:
    int index;            // the row-major index of this agent
    int size;             // the size of the Agent array
    Place *place;         //Points to the current place where this agent resides.
    unsigned placeIndex;  //index of the place where agent resides
    unsigned placeDevIndex; // REMOVE?
    int myDevice;
    int placeAgentArrayIdx; // index of this Agent in the Place's array for which it resides
    bool isAlive;
    bool agentTraveled;
    bool isAccepted;
    int traveledAgentIdx;
    bool longDistanceMigration; // REFACTOR TO REMOVE
    Place *destPlace;
    int destPlaceIdx;
    bool markForDelete;

    int nChildren;  //number of agents to spawn at the next call to migrate()
    Place *childPlace;

};

} /* namespace mass */

#endif /* AGENTSTATE_H_ */
