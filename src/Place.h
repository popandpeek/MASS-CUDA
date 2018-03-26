
#pragma once

// easier for end users to understand than the __host__ __device__ meaning.
#define MASS_FUNCTION __host__ __device__

#include<cuda_runtime.h>
#include "Agent.h"
#include "settings.h"

namespace mass {

// forward declaration
class PlaceState;
//class Agent;

/**
 *  The Place class defines the default functions for acheiving GPU parallelism between place objects.
 *  It also defines the interface necessary for end users to implement.
 */
class Place {

public:
	/**
	 *  A contiguous space of arguments is passed 
	 *  to the constructor.
	 */
	MASS_FUNCTION Place(PlaceState* state, void *args = NULL);

	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL) = 0;

	//MASS_FUNCTION virtual void setState(PlaceState *s);

	MASS_FUNCTION virtual PlaceState* getState();

	/* The default migration conflict resolution is based on the ID of the migrating agent.
	Can be overloaded in the derived Agent class
	*/
	MASS_FUNCTION virtual void resolveMigrationConflicts();

	MASS_FUNCTION int getIndex();

	MASS_FUNCTION void setIndex(int index);

	MASS_FUNCTION void setSize(int *dimensions, int nDims);

	MASS_FUNCTION bool addAgent(Agent* agent);

	MASS_FUNCTION void removeAgent(Agent* agent);

	MASS_FUNCTION int getAgentPopulation();

	MASS_FUNCTION void addMigratingAgent(Agent* agent, int relativeIdx);


	PlaceState *state;

};
} /* namespace mass */
