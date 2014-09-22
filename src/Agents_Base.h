/**
 *  @file Agents_Base.h
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
#include "Places_Base.h"

namespace mass {
class Dispatcher;

class Agents_Base {
	friend class Dispatcher;

public:

	virtual ~Agents_Base();

	virtual int getHandle();

	virtual int nAgents();

	virtual void callAll(int functionId) = 0;

	virtual void callAll(int functionId, void *argument, int argSize) = 0;

	virtual void *callAll(int functionId, void *arguments[], int argSize, int retSize) = 0;

	virtual void manageAll() = 0;

protected:
	// Agent creation is handled through Mass::createAgents(...) call
	Agents_Base(int handle, void *argument, int argument_size, Places_Base *places,
			int initPopulation);

	Places_Base *places; /**< The places used in this simulation. */
	int handle; /**< Identifies the type of agent this is.*/
  void *argument;
  int argSize;
	int numAgents; /**< Running count of living agents in system.*/
	int newChildren; /**< Added to numAgents and reset to 0 each manageAll*/
	int sequenceNum; /*!< The number of agents created overall. Used for agentId creation. */
	Dispatcher *dispatcher;
};
// end class

}// mass namespace

//#endif // AGENTS_H_
