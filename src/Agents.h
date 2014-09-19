/**
 *  @file Agents.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <string>
#include <vector>
#include "Agent.h"
#include "Dispatcher.h"
#include "Model.h"
#include "Places.h"

namespace mass {

class Agents {
	friend class Model;
	friend class Dispatcher;

public:

//	Agents(int handle, std::string className, void *argument, int argSize,
//			Places<Place> *places, int initPopulation){
//		this->places = places;
//		this->agents = NULL;
//		this->handle = handle;
//		this->numAgents = initPopulation;
//		this->newChildren = initPopulation;
//		this->sequenceNum = 0;
//	}

	~Agents(){
		if(NULL != agents){
			delete[] agents;
			agents = NULL;
		}
	}

	int getHandle(){
		return handle;
	}

	int nAgents(){
		return numAgents;
	}

	void callAll(int functionId){
		//TODO send call all command to dispatcher
	}

	void callAll(int functionId, void *argument, int argSize){
		//TODO send call all command to dispatcher
	}

	void *callAll(int functionId, void *arguments[], int argSize, int retSize){
		//TODO send call all command to dispatcher
	}

	void manageAll(){
		//TODO send manage all command to dispatcher
	}

private:

	Places *places; /**< The places used in this simulation. */

	Agent* agents; /**< The agents elements.*/

	int handle; /**< Identifies the type of agent this is.*/

	int numAgents; /**< Running count of living agents in system.*/

	int newChildren; /**< Added to numAgents and reset to 0 each manageAll*/

	int sequenceNum; /*!< The number of agents created overall. Used for agentId creation. */
};
// end class

}// mass namespace

//#endif // AGENTS_H_
