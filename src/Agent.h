#ifndef AGENT_H
#define AGENT_H

#pragma once
#define MASS_FUNCTION __host__ __device__
#include<cuda_runtime.h>

namespace mass {

// forward declaration
class AgentState;
class Place;

class Agent {

public:
    
    /**
    The default constructor. A contiguous space of arguments is 
    passed to the constructor of the derived classes.
    */
    MASS_FUNCTION Agent(AgentState* state, void *args = NULL);

    /**
    Called by MASS while executing Agents.callAll(). This is intended 
    to be a switch statement where each user-implemented function is 
    mapped to a funcID, and is passed ‘args’ when called.
    */
    MASS_FUNCTION virtual void callMethod(int, void *arg = NULL) = 0;

    /**
    Returns the AgentState object pointed associated with this Agent
    */
    MASS_FUNCTION virtual AgentState* getState();

    /**
    Assigns this Agents to a particular Place (it’s place of residency)
    */
    MASS_FUNCTION void setPlace(Place*);

    /**
    Returns a pointer to the Place, on which this agent is located/resides. 
    This function can only be called from the GPU code. 
    */
    MASS_FUNCTION Place* getPlace();

    /**
    Returns the index to the Place, on which this agent is located/resides. 
    This function can be called from the CPU and GPU code.
    */
    MASS_FUNCTION int getPlaceIndex();

    MASS_FUNCTION int getPlaceDevIndex();

    MASS_FUNCTION void setMyDevice(int);

    MASS_FUNCTION int getMyDevice();

    /**
    Returns the unique index of this Agent
    */
    MASS_FUNCTION int getIndex();

    /**
    Sets the unique index of this Agent
    */
    MASS_FUNCTION void setIndex(int);

    MASS_FUNCTION void setDestPlace(Place*);

    MASS_FUNCTION void setDestPlaceIndex(int);

    MASS_FUNCTION Place* getDestPlace();
    
    MASS_FUNCTION int getDestPlaceIndex();
    /**
    Returns true if the Agent is in an active state. Returns false if the 
    agent hasn’t been activated yet (extra inactive agents are allocated 
    at simulation start to optimize memory allocation), or if it has 
    already been terminated.
    */
    MASS_FUNCTION bool isAlive();

    /**
    Sets the Agent’s status to active, thus including Agent into all callAll() calls.
    */
    MASS_FUNCTION void setAlive(bool);

    /**
    Sets the traveled status of Agent -> True for having traveled devices, false for not
    */
    MASS_FUNCTION void setTraveled(bool);

    MASS_FUNCTION bool isTraveled();  

    MASS_FUNCTION void setTraveledAgentIdx(int);

    MASS_FUNCTION int getTraveledAgentIdx();

    MASS_FUNCTION void setAccepted(bool);

    MASS_FUNCTION bool isAccepted();

    MASS_FUNCTION bool longDistanceMigration();
    
    MASS_FUNCTION void setLongDistanceMigration(bool);

    MASS_FUNCTION void setPlaceAgentArrayIdx(int);

    MASS_FUNCTION int getPlaceAgentArrayIdx();

    MASS_FUNCTION void markForTermination(bool);
    /**
    Sets the agent status to inactive. Agents place is set to vacant and 
    he agent is excluded from all the subsequent callAll() function calls.
    */
    MASS_FUNCTION void terminateAgent();

    MASS_FUNCTION void terminateGhostAgent();

    MASS_FUNCTION void markAgentForTermination();

    MASS_FUNCTION bool isMarkedForTermination();
    /**
    Moves the agent from the current place to the destination place. 
    The actual migration is performed during the subsequent Agents::manageAll() call. 
    The migration is not guaranteed is the destination place is already occupied 
    or there is a competing agent trying to get to the same destination. 
    @param destinationRelativeIdx is the index of the destination Place in the 
    migration destination array. This parameter can have an integer value from 
    0 to N_DESTINATIONS-1, where N_DESTINATIONS is defined in the src/settings.h file. 
    This parameter is required to save different agents migrating to a place from 
    different surrounding locations into separate places in memory.
    */
    MASS_FUNCTION void migrateAgent(Place*, int);

    MASS_FUNCTION void migrateAgentLongDistance(Place*, int);
    /**
    Spawns the specified number of new agents at the specified place.
    The function does not accept instantiation arguments due to current limitations of the library.
    If you need to set some parameters of the newly created Agents, please set these parameters 
    in a separate function after agent creation.
    */
    MASS_FUNCTION void spawn(int, Place*);

    AgentState *state;
};
}
#endif