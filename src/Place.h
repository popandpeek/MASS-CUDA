/**
 *  @file Place.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

// change either of these numbers to optomize for a particular simulation's needs
#define MAX_AGENTS 4
#define MAX_NEIGHBORS 8
#define MAX_DIMS 6

#include<cuda_runtime.h>

#include "MObject.h"

namespace mass {

// forward declaration
class Agent;
class PlaceState;

/**
 *  The Place class defines the default functions for acheiving GPU parallelism between place objects.
 *  It also defines the interface necessary for end users to implement.
 */
class Place: public MObject {
	friend class Agent;
	friend class Places;

public:
	/**
	 *  A contiguous space of arguments is passed 
	 *  to the constructor.
	 */
	MASS_FUNCTION Place(void *args = NULL);

//	/**
//	 *  Called by MASS while executing Places.callAll().
//	 *
//	 * @param functionId user-defined function id
//	 * @param args user-defined arguments
//	 */
//	MASS_FUNCTION virtual void callMethod(int functionId, void* args) = 0;

	/**
	 *  Gets a pointer to this place's out message.
	 */
	MASS_FUNCTION virtual void *getMessage() = 0;

	/**
	 * Returns the number of bytes necessary to store this agent implementation.
	 * The most simple implementation is a single line of code:
	 * return sizeof(ThisClass);
	 *
	 * Because sizeof is respoved at compile-time, the user must implement this
	 * function rather than inheriting it.
	 *
	 * @return an int >= 0;
	 */
	MASS_FUNCTION virtual int placeSize() = 0;

	/**
	 * Registers an agent with this place.
	 * @param agent the agent that is self-registering.
	 */
	MASS_FUNCTION void addAgent(Agent *agent);

	/**
	 * Unregisters an agent with this place.
	 * @param agent the agent that is self-unregistering.
	 */
	MASS_FUNCTION void removeAgent(Agent *agent);


	// TODO remove this call if not necessary
	MASS_FUNCTION virtual void callMethod( int functionId, void *arg = NULL) = 0;

	MASS_FUNCTION virtual void setState(PlaceState *s);

	MASS_FUNCTION virtual PlaceState* getState();

	MASS_FUNCTION int getIndex();

protected:

//	int size[MAX_DIMS];   // the size of the Places matrix
//	char numDims;
//	int index;            // the row-major index of this place
//	Place *neighbors[MAX_NEIGHBORS];  // my neighbors
//	Agent *agents[MAX_AGENTS];
//	unsigned agentPop; // the population of agents on this place
//	// void* outMessage;        // out message needs to be declared in the derived class statically
//	int message_size;  // the number of bytes in a message
//	void *inMessages[MAX_NEIGHBORS]; // holds a pointer to each neighbor's outmessage.
	PlaceState *state;

};
} /* namespace mass */
