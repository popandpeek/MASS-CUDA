/**
 *  @file Agents.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <map>
#include <string>
#include <vector>

#include "Agent.h"
#include "AgentsPartition.h"
#include "Dispatcher.h"

// forward declarations
class Places;

namespace mass {

class Agents {
    friend class AgentsPartition;
    friend class Dispatcher;

public:

    virtual ~Agents ( );

    int getHandle ( );

    int getPlacesHandle ( );

    int nAgents ( );

    void callAll ( int functionId );

    void callAll ( int functionId, void *argument, int argSize );

    void *callAll ( int functionId, void *arguments[ ], int argSize, int retSize );

    void manageAll ( );

    int getNumPartitions ( );


protected:
	// Agent creation is handled through Mass::createAgents(...) call
    Agents ( int handle, void *argument, int argument_size, Places *places,
             int initPopulation );

    
    void addPartitions ( std::vector<AgentsPartition*> parts );

    AgentsPartition *getPartition ( int rank );
    
    void setTsize ( int size );

    int getTsize ( );

    Places *places; /**< The places used in this simulation. */
    int handle; /**< Identifies the type of agent this is.*/
    void *argument;
    int argSize;
    int numAgents; /**< Running count of living agents in system.*/
    int newChildren; /**< Added to numAgents and reset to 0 each manageAll*/
    int sequenceNum; /*!< The number of agents created overall. Used for agentId creation. */
    Dispatcher *dispatcher;
    int Tsize;
    Agent **agentPtrs;
	std::map<int, AgentsPartition*> partitions;
};// end class

}// mass namespace
