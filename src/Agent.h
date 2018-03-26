#pragma once
#define MASS_FUNCTION __host__ __device__
#include<cuda_runtime.h>

namespace mass {

// forward declaration
class AgentState;
class Place;

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

    MASS_FUNCTION void setPlace(Place* place);
    MASS_FUNCTION Place* getPlace();
    MASS_FUNCTION int getPlaceIndex();

    MASS_FUNCTION int getIndex();

    MASS_FUNCTION void setIndex(int index);

    MASS_FUNCTION bool isAlive();
    MASS_FUNCTION void setAlive();
    MASS_FUNCTION void terminateAgent();

    MASS_FUNCTION void migrateAgent(Place* destination, int destinationRelativeIdx);

    /* Spawns the specified number of new agents at the specified place.
    The function does not accept instantiation arguments due to current limitations of the library.
    If you need to set some parameters of the newly created Agents, please set these parameters 
    in a seprate function after agent creation.
    */
    MASS_FUNCTION void spawn(int numAgents, Place* place);

    AgentState *state;
};
}