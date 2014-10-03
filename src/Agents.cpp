/**
 *  @file Agents.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Agents.h"
#include "AgentsPartition.h"
#include "Mass.h"

namespace mass {

    Agents::~Agents ( ) {
        if ( NULL != agentPtrs ) {
            delete[ ] agentPtrs;
        }
        partitions.empty ( );
    }

    int Agents::getHandle ( ) {
        return handle;
    }

    int Agents::getPlacesHandle ( ) {
        return places->getHandle ( );
    }

    int Agents::nAgents ( ) {
        return numAgents;
    }

    void Agents::callAll ( int functionId ) {
        callAll ( functionId, NULL, 0 );
    }

    void Agents::callAll ( int functionId, void *argument, int argSize ) {
        dispatcher->callAllAgents ( this->getHandle(), functionId, argument, argSize );
    }

    void *Agents::callAll ( int functionId, void *arguments[ ], int argSize,
                            int retSize ) {
        return dispatcher->callAllAgents ( this->getHandle(), functionId, arguments, argSize, retSize );
    }

    void Agents::manageAll ( ) {
        dispatcher->manageAllAgents ( this->getHandle() );
    }

    int Agents::getNumPartitions ( ) {
        return partitions.size ( );
    }

    Agents::Agents ( int handle, void *argument, int argument_size, Places *places,
                     int initPopulation ) {

        this->places = places;
        this->handle = handle;
        this->argument = argument;
        this->argSize = argument_size;
        this->numAgents = initPopulation;
        this->newChildren = 0;
        this->sequenceNum = 0;
        this->dispatcher = Mass::dispatcher;
        this->Tsize = 0;
        this->agentPtrs = NULL;
    }

    void Agents::addPartitions ( std::vector<AgentsPartition*> parts ) {
        int totalQty = 0;
        for ( int i = 0; i < parts.size ( ); ++i ) {
            totalQty += parts[ i ]->size ( );
        }

        // remove any old pointers
        if ( NULL != agentPtrs ) {
            delete[ ] agentPtrs;
        }

        agentPtrs = new Agent*[ totalQty ];

        int nextPtr = 0;
        // set pointers to new agents
        for ( int i = 0; i < parts.size ( ); ++i ) {
            AgentsPartition* part = parts[ i ];
            int qty = part->size ( );
            // allow pointer arithmatic in bytes
            char* nextAgent = ( char* ) part->hostPtr ( );
            for ( int j = 0; j < qty; ++j ) {
                agentPtrs[ nextPtr++ ] = ( Agent* ) nextAgent;
                nextAgent += Tsize; // moves pointer to next agent's memory address
            }
        }
    }

    AgentsPartition *Agents::getPartition ( int rank ) {
        return partitions[ rank ];
    }


    void Agents::setTsize ( int size ) {
        Tsize = size;
    }

    int Agents::getTsize ( ) {
        return Tsize;
    }
}// mass namespace
