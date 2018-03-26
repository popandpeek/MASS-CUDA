
#ifndef ANTSTATE_H_
#define ANTSTATE_H_

#include "../src/AgentState.h"

class AntState: public mass::AgentState {
public:
    int agentSugar, agentMetabolism, destinationIdx;
};

#endif /* ANTSTATE_H_ */
