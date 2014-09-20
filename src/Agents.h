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
#include "Model.h"
#include "Places.h"

namespace mass {
class Dispatcher;

class Agents {
	friend class Model;
	friend class Dispatcher;

public:

	~Agents();

	int getHandle();

	int nAgents();

	void callAll(int functionId);

	void callAll(int functionId, void *argument, int argSize);

	void *callAll(int functionId, void *arguments[], int argSize, int retSize);

	void manageAll();

private:
	// Agent creation is handled through Mass::createAgents(...) call
	Agents(int handle, void *argument, int argument_size, Places *places,
			int initPopulation);

	Places *places; /**< The places used in this simulation. */

	Agent* agents; /**< The agents elements.*/

	int handle; /**< Identifies the type of agent this is.*/

	int numAgents; /**< Running count of living agents in system.*/

	int newChildren; /**< Added to numAgents and reset to 0 each manageAll*/

	int sequenceNum; /*!< The number of agents created overall. Used for agentId creation. */

	Dispatcher *dispatcher;
};
// end class

}// mass namespace

//#endif // AGENTS_H_
