/**
 *  @file Place.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Place.h"
#include "Agent.h"

namespace mass {

/**
 *  A contiguous space of arguments is passed
 *  to the constructor.
 */
__host__ __device__ Place::Place(void *args) {
	agentPop = 0;
}

/**
 * Registers an agent with this place.
 * @param agent the agent that is self-registering.
 */
__host__ __device__ void Place::addAgent(Agent *agent) {
	// this works because of unique migration pattern that prevents collisions.
	unsigned idx = agentPop++;
	if (idx >= MAXAGENTS) {
		--agentPop; // TODO silent failure is a shitty way to deal with this
	} else {
		agent->placePos = idx;
		agents[idx] = agent;
	}
}

/**
 * Unregisters an agent with this place.
 * @param agent the agent that is self-unregistering.
 */
__host__ __device__ void Place::removeAgent(Agent *agent) {
	unsigned idx = agent->placePos;
	agents[idx] = NULL;
	--agentPop;
}

//	int *size;            // the size of the Places matrix
//	int index;            // the row-major index of this place
//	Place *neighbors[MAXNEIGHBORS];  // my neighbors
//	Agent *agents[MAXAGENTS];
//	unsigned agentPop;
//	// void* outMessage;        // out message needs to be declared in the derived class statically
//	int outMessage_size;  // the number of bytes in an out message
//	void *inMessages[MAXNEIGHBORS]; // holds a pointer to each neighbor's outmessage.
//	int inMessage_size; // the size of an in message

} /* namespace mass */

