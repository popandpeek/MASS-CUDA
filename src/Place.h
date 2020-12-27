
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
	Default constructor
	*/
	MASS_FUNCTION Place();
	
	/**
	The constructor called on the GPU device.
	A contiguous space of arguments is passed to the constructor.
	*/
	MASS_FUNCTION Place(PlaceState* state, void *args = NULL);
	
	/**
	Called by MASS while executing Places.callAll(). 
	This is intended to be a switch statement where each user-implemented 
	function is mapped to a funcID, and is passed ‘args’ when called.
	*/
	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL) = 0;

	/**
	Sets state pointer
	*/
	MASS_FUNCTION void setState(PlaceState *pState) ;

	/**
	Returns the PlaceState object pointed associated with this Place
	*/
	MASS_FUNCTION virtual PlaceState* getState();

	/**
	The default migration conflict resolution algorithm selects the agent 
	with the lowest index, in case several agents intend to migrate to the 
	same place. This method can be overloaded in the derived Place class if 
	the application requires a different conflict resolution mechanism.
	*/
	MASS_FUNCTION virtual void resolveMigrationConflicts();

	/**
	Returns the index of this Place.
	*/
	MASS_FUNCTION int getIndex();

	/**
	Return relative index of this place
	*/
	MASS_FUNCTION int getDevIndex();
	/**
	Sets the index of this place to the specified integer value.
	*/
	MASS_FUNCTION void setIndex(int index);

	/**
	Sets the index of this place relative to the entire space of Place objects
	*/
	MASS_FUNCTION void setDevIndex(int devIndex);
	/**
	Adds a resident Agent to this place. Returns true if the adding was successful. 
	Adding can fail if the place has reached its maximum occupancy.
	*/
	MASS_FUNCTION bool addAgent(Agent* agent);

	/**
	Removes the specified Agent from the array of resident Agents in this place.
	*/
	MASS_FUNCTION void removeAgent(Agent* agent);

	/**
	Returns the number of Agents currently residing in this place.
	*/
	MASS_FUNCTION int getAgentPopulation();

	/**
	This is a utility function used within the library. 
	Adds an agent to an array of agents who expressed an intention to 
	migrate to this place. The agents which will be actually accepted 
	for migration by a place are identified during the Agents::manageAll() call. 
	@param relativeIdx is the index of the destination Place in the migration 
	destination array. This parameter can have an integer value from 0 to 
	N_DESTINATIONS-1, where N_DESTINATIONS is defined in the src/settings.h file. 
	This parameter is required to save different agents migrating to a place 
	from different surrounding locations into separate places in memory.
	*/
	MASS_FUNCTION void addMigratingAgent(Agent* agent, int relativeIdx);


	MASS_FUNCTION void setSize(int *placesDimensions, int *devDimensions, int nDims);
	PlaceState *state;

};
} /* namespace mass */
