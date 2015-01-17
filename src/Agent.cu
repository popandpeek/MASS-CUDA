/**
 *  @file Agent.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Agent.h"
#include "Place.h"

namespace mass {

//unsigned alive; // 0 is dead
//Place *place;
//int placeHandle;
//int index;
//int destIndex;
//int agentId;
//int parented;
//int newChildren;
//void* arguments;
//int argSize;
//int agentsHandle;

/**
 *  \brief Create an agent with user defined values to initialize fields.
 *  \details This is the preferred constructor, as it is unnecessary to
 *  call init() afterwards.
 *
 *  @param [in] args Parameter_Description
 */__host__ __device__ Agent::Agent(void * args) {
	// abstract class does not store args
}

Agent::~Agent() {
	// no dynamic memory management here
}

__host__ __device__ int Agent::map(int initPopulation, int *size, int index) {
	// get the number of places
	int nDim = sizeof(size) / sizeof(size[0]);
	int numPlaces = 1;
	for (int i = 0; i < nDim; ++i) {
		numPlaces *= size[i];
	}

	// determine number of agents at every place
	int numAgents = initPopulation / numPlaces;

	// number of agents that need to be distributed evenly
	int remainder = initPopulation % numPlaces;

	// if this index is evenly divisible by spacing, add an agent
	if (remainder != 0 && (index % (numPlaces / remainder)) == 0) {
		if (remainder > 1 && index != 0) // special case where extra agent is added
			++numAgents;
	}

	return numAgents;
}

/**
 *  Terminates the calling agent upon a next call to Agents.manageAll( ).
 *  More specifically, kill( ) sets the “alive” variable false.
 */__device__ void Agent::kill() {
	alive = 0;
}

/**
 *	Checks to see if this agent is alive
 *
 *	@return true if agent is alive
 */__device__ bool Agent::isAlive() {
	return 0 != alive;
}

/**
 *	Sets this Agent's place pointer to the provided pointer.
 *
 *	@param placePtr a pointer to the place where this agent resides.
 */__device__ void Agent::setPlace(Place *placePtr) {
	place = placePtr;
	index = place->getIndex();
}

/**
 *   Initiates an agent migration upon a next call to Agents.manageAll( ).
 *   More specifically, migrate( ) updates the calling agent’s index[].
 *
 *   @param row-major index the destination Place of this migration.
 *   @return <code>true</code> if a migration will occur.
 */__device__ bool Agent::migrate(int index) {
	destIndex = index;
	return destIndex != this->index;
}

/**
 *  Spawns a “numAgents’ of new agents, as passing arguments[i] (with
 *  arg_size) to the i-th new agent upon a next call to Agents.manageAll( ).
 *  More specifically, spawn( ) changes the calling agent’s newChildren.
 *
 *  @param numAgents the number of new agents to spawn.
 *  @param arguments the arguments used to create the spawned agents.
 *  @param argSize the number of arguments in the arguments array
 */__device__ void Agent::spawn(int numAgents, void* arguments, int argSize) {
	if (numAgents > 0) {
		this->newChildren = numAgents;
		this->arguments = arguments;
		this->argSize = argSize;
	}
}

} // namespace mass
