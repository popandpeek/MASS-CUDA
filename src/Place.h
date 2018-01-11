
#pragma once

// change either of these numbers to optomize for a particular simulation's needs
#define MAX_AGENTS 4
#define MAX_NEIGHBORS 8
#define MAX_DIMS 6
 // easier for end users to understand than the __host__ __device__ meaning.
#define MASS_FUNCTION __host__ __device__

#include<cuda_runtime.h>

namespace mass {

// forward declaration
class PlaceState;

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
	 * Because sizeof is resolved at compile-time, the user must implement this
	 * function rather than inheriting it.
	 *
	 * @return an int >= 0;
	 */
	//MASS_FUNCTION virtual int placeSize() = 0;

	// TODO remove this call if not necessary
	MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL) = 0;

	//MASS_FUNCTION virtual void setState(PlaceState *s);

	MASS_FUNCTION virtual PlaceState* getState();

	MASS_FUNCTION int getIndex();

	MASS_FUNCTION void setIndex(int index);

	MASS_FUNCTION void setSize(int *dimensions, int nDims);


	PlaceState *state;

};
} /* namespace mass */
