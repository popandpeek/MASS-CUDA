#pragma once
#define MASS_FUNCTION __host__ __device__
#include<cuda_runtime.h>

namespace mass {

// forward declaration
class AgentState;

class Agent {

public:

    // map(int maxAgents, vector<int> size, vector<int> coordinates ); - > from c++ lib

    /**
     *  A contiguous space of arguments is passed 
     *  to the constructor.
     */
    MASS_FUNCTION Agent(AgentState* state, void *args = NULL);

    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL) = 0;

    MASS_FUNCTION virtual AgentState* getState();

    MASS_FUNCTION int getIndex();

    MASS_FUNCTION void setIndex(int index);

    MASS_FUNCTION void setSize(int qty);


    AgentState *state;

};
}