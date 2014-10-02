/**
 *  @file Place.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

// change either of these numbers to optomize for a particular simulation's needs
#define MAXAGENTS 4
#define MAXNEIGHBORS 8

#include<cuda_runtime.h>
#include "Agent.h"

namespace mass {

/**
 *  The Place class defines the default functions for acheiving GPU parallelism between place objects.
 *  It also defines the interface necessary for end users to implement.
 */
class Place {
	friend class Agent;

public:
	/**
	 *  A contiguous space of arguments is passed 
	 *  to the constructor.
	 */
	__host__ __device__ Place(void *args);

	/** 
	 *  Called by MASS while executing Places.callAll().
	 *
	 * @param functionId user-defined function id
	 * @param args user-defined arguments
	 */
	__device__ virtual void callMethod(int functionId, void* args) = 0;

	/**
	 *  Gets a pointer to this place's out message.
	 */
	__host__ __device__ virtual void *getMessage() = 0;

	/**
	 * Registers an agent with this place.
	 * @param agent the agent that is self-registering.
	 */
	__host__ __device__ void addAgent(Agent *agent);

	/**
	 * Unregisters an agent with this place.
	 * @param agent the agent that is self-unregistering.
	 */
	__host__ __device__ void removeAgent(Agent *agent);

protected:

	int *size;            // the size of the Places matrix
	int index;            // the row-major index of this place
	Place *neighbors[MAXNEIGHBORS];  // my neighbors
	Agent *agents[MAXAGENTS];
	unsigned agentPop; // the population of agents on this place
	// void* outMessage;        // out message needs to be declared in the derived class statically
	int outMessage_size;  // the number of bytes in an out message
	void *inMessages[MAXNEIGHBORS]; // holds a pointer to each neighbor's outmessage.
	int inMessage_size; // the size of an in message

};
} /* namespace mass */
