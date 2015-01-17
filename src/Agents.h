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

namespace mass {

// forward declarations
class Agent;
class Dispatcher;
class Places;

class Agents {
	friend class Mass;

public:

	virtual ~Agents();

	int getHandle();

	int getPlacesHandle();

	Agent** getAgentElements();

	int nAgents();

	void callAll(int functionId);

	void callAll(int functionId, void *argument, int argSize);

	void *callAll(int functionId, void *arguments[], int argSize, int retSize);

	void manageAll();

protected:
	// Agent creation is handled through Mass::createAgents(...) call
	Agents(int handle, void *argument, int argument_size, Places *places,
			int initPopulation);

	void setDispatcher(Dispatcher *d);

	Places *places; /**< The places used in this simulation. */
	int handle; /**< Identifies the type of agent this is.*/
	int numAgents; /**< Running count of living agents in system.*/
	Dispatcher *dispatcher;
	Agent **agentPtrs;

};
// end class

}// mass namespace
