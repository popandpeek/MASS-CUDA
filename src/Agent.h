/**
 *  @file Agent.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
//#ifndef AGENT_H_
//#define AGENT_H_
#pragma once
#include<cuda_runtime.h>
#include <string>

namespace mass {
class Place;
/**
 *  This is the basic Agent class. It provides core functionality and should be
 *  used as the parent class for any user-defined agents. Virtual functions may
 *  be overridden as necessary,
 */
class Agent {
	friend class Place;

public:

	/*******************************************************************************
	 *  USERS MUST OVERRIDE THESE ABSTRACT FUNCTIONS.
	 ******************************************************************************/

	/**
	 *	Gets a user defined value from this agent.
	 */
	virtual void* getValue() = 0;

	/**
	 *  Is called from Agents.callAll. 
	 *  It invokes the function specified with functionId as passing arguments to 
	 *  this function. A user-derived Agent class must implement this method. A 
	 *  typical implementation uses a switch statement on functionId to parse 
	 *  arguments and call a specific function.
	 *
	 *  @param functionId the id of a user specified function.
	 *  @param arguments the arguments for that function.
	 */
	__device__ virtual void* callMethod(int functionId, void *arguments) = 0;

	/**
	 *  Is called from Agents.callAll. It invokes the function specified with
	 *  functionId. Simply call callMethod(functionId, arguments) with
	 *	NULL as the arguments.
	 *
	 *  @param functionId the id of a user specified function.
	 */
	__device__ virtual void* callMethod(int functionId) = 0;

	/*******************************************************************************
	 *  END ABSTRACT FUNCTIONS.
	 *	The following functions may be overridden as desired, although it is not 
	 *	necessary for most applications.
	 ******************************************************************************/

	/**
	 *  \brief Create an agent with user defined values to initialize fields.
	 *  \details This is the preferred constructor, as it is unnecessary to 
	 *  call init() afterwards.
	 *  
	 *  @param [in] args Parameter_Description
	 */
	__host__ __device__ Agent(void * args);

	/**
	 *  The destructor.
	 */
	~Agent();

	/**
	 *  Returns the number of agents to initially instantiate on a place indexed
	 *  with coordinates[]. The system-provided (thus default) map( ) method
	 *  distributes agents over places uniformly as in:
	 *
	 *      maxAgents / size.length
	 *
	 *  The map( ) method may be overloaded by an application-specific method.
	 *  A user-provided map( ) method may ignore maxAgents when creating agents.
	 *
	 *  @param initPopulation indicates the number of agents to create over the entire
	 *                   application
	 *  @param size defines the size of the Places matrix to which a given Agent
	 *              class belongs.
	 *  @param index the index of the place where agents might be created.
	 *	@return the number of agents to create at the place specified by index
	 */
	__host__ __device__ virtual int map(int initPopulation, int *size,
			int index);

	/**
	 *  Terminates the calling agent upon a next call to Agents.manageAll( ).
	 *  More specifically, kill( ) sets the “alive” variable false.
	 */
	__device__ virtual void kill();

	/**
	 *	Checks to see if this agent is alive
	 *
	 *	@return true if agent is alive
	 */
	__device__ bool isAlive();

	/**
	 *	Sets this Agent's place pointer to the provided pointer.
	 *
	 *	@param placePtr a pointer to the place where this agent resides.
	 */
	__device__ void setPlace(Place *placePtr);

	/**
	 *  Is 1 while this agent is active. Once it is set 0, this agent is
	 *  considered inactive.
	 */
	unsigned alive;

protected:

	/**
	 *  Points to the current place where this agent resides.
	 */
	Place *place;

	/**
	 * Stores the index where this Agent is stored in a single Place's agent pointer array.
	 */
	unsigned placePos;

	/**
	 *  Maintains the handle of the places class with which this agent is
	 *  associated.
	 */
	int placeHandle;

	/**
	 *  The row-major index of the place where this agent resides.
	 */
	int index;

	/**
	 *	This is the migration destination. If destIdx != index,
	 *	this agent will move on the next call to manageAll().
	 */
	int destIndex;

	/**
	 *  Is this agent’s identifier. It is calculated as: the sequence number * the
	 *  size of this agent’s belonging matrix + the index of the current place
	 *  when all places are flattened to a single dimensional array.
	 */
	int agentId;

	/**
	 *  Is the identifier of this agent’s parent.
	 */
	int parented;

	/**
	 *  Is the number of new children created by this agent upon a next call to
	 *  Agents.manageAll( ).
	 */
	int newChildren;

	/**
	 *  Is an array of arguments, each passed to a different new child.
	 */
	void* arguments;

	/**
	 * Is the size of the arguments array.
	 */
	int argSize;

	/**
	 *  Maintains the handle of the agents class to which this agent belongs.
	 */
	int agentsHandle;

	/**
	 *   Initiates an agent migration upon a next call to Agents.manageAll( ).
	 *   More specifically, migrate( ) updates the calling agent’s index[].
	 *
	 *   @param row-major index the destination Place of this migration.
	 *   @return <code>true</code> if the migration occurred.
	 */
	__device__ bool migrate(int index);

	/**
	 *  Spawns a “numAgents’ of new agents, as passing arguments[i] (with
	 *  arg_size) to the i-th new agent upon a next call to Agents.manageAll( ).
	 *  More specifically, spawn( ) changes the calling agent’s newChildren.
	 *
	 *  @param numAgents the number of new agents to spawn.
	 *  @param arguments the arguments used to create the spawned agents.
	 *  @param argSize the number of arguments in the arguments array
	 */
	__device__ void spawn(int numAgents, void* arguments, int argSize);

};
// class Agent

}// namespace mass

//#endif // AGENT_H_
