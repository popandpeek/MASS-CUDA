
#ifndef AGENTSTATE_H_
#define AGENTSTATE_H_

#include "Place.h"

namespace mass {

class AgentState {
    friend class Agent;

public:
    int index;            // the row-major index of this place
    int size;             // the size of the Agent array
    Place *place;         //Points to the current place where this agent resides.
    unsigned placeIndex;  //index of the place where agent resides
    bool isAlive;

    Place *destPlace;

    int nChildren;  //number of agents to spawn at the next call to migrate()
    Place *childPlace;

};

} /* namespace mass */

#endif /* AGENTSTATE_H_ */