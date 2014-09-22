/**
 *  @file Agents.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include "Agents_Base.h"
#include "AgentsPartition.h"
#include "Dispatcher.h"

namespace mass {

template<typename T>
class Agents : public Agents_Base {

public:

	virtual ~Agents() {
		agents.clear();
		partitions.empty();
	}

	virtual void callAll(int functionId) {
		callAll(functionId, NULL, 0);
	}

	virtual void callAll(int functionId, void *argument, int argSize) {
		dispatcher->callAllAgents < T > (handle, functionId, argument, argSize);
	}

	virtual void *callAll(int functionId, void *arguments[], int argSize,
			int retSize) {
		return dispatcher->callAllAgents < T
				> (handle, functionId, arguments, argSize, retSize);
	}

	virtual void manageAll() {
		dispatcher->manageAllAgents < T > (handle);
	}

	virtual int getNumPartitions(){
		return agents.size();
	}

protected:
	// Agent creation is handled through Mass::createAgents(...) call
	Agents(int handle, void *argument, int argument_size, Places_Base *places,
			int initPopulation) :
			Agents_Base(handle, argument, argument_size, places, initPopulation) {
	}

	std::vector<T*> agents; /**< The agents elements.*/
	std::map<int, AgentsPartition<T>*> partitions;
};
// end class

}// mass namespace

//#endif // AGENTS_H_
