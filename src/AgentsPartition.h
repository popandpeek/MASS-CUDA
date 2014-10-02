/**
 *  @file AgentsPartition.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <string>
#include <vector>
#include "Agents.h"


namespace mass {

class AgentsPartition {

public:

    AgentsPartition ( int handle, void *argument, int argument_size, Agents *agents,
                      int numElements );

	/**
	 *  Destructor
	 */
    ~AgentsPartition ( );

	/**
	 *  Returns the number of elements in this partition.
	 */
    int size ( );

	/**
	 *  Returns the number of elements and ghost elements.
	 */
    int sizePlusGhosts ( );

	/**
	 *  Gets the rank of this partition.
	 */
    int getRank ( );

	/**
	 *  Returns an array of the elements contained in this Partition.
	 */
    void *hostPtr ( );

	/**
	 *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
	 */
    void *hostPtrPlusGhosts ( );

	/**
	 *  Returns the pointer to the GPU data. NULL if not on GPU.
	 */
    void *devicePtr ( );

    void setDevicePtr ( void *agents );

	/**
	 *  Returns the handle associated with this AgentsPartition object that was set at construction.
	 */
    int getHandle ( );

	/**
	 *  Sets the start and number of agents in this partition.
	 */
    void setSection ( void *start );

    void setQty ( int qty );

    bool isLoaded ( );

    bool setLoaded ( bool loaded );

    void makeLoadable ( );

    void *load ( cudaStream_t stream );

    bool retrieve ( cudaStream_t stream, bool freeOnRetrieve );

    int getGhostWidth ( );

    void setGhostWidth ( int width, int n, int *dimensions );

	// TODO Do these ghost updates do what I want?
	// look at having them move the data to a destination pointer
    void updateLeftGhost ( void *ghost, cudaStream_t stream );

    void updateRightGhost ( void *ghost, cudaStream_t stream );

	// TODO add a pointer param so buffers and ghosts can be copied directly where they need to go
    void *getLeftBuffer ( );

    void *getRightBuffer ( );

    dim3 blockDim ( );

    dim3 threadDim ( );

    void setIdealDims ( );

    int getPlaceBytes ( );

private:
	void *hPtr; // this starts at the left ghost, and extends to the end of the right ghost
	void *dPtr; // pointer to GPU data
	int handle;         // User-defined identifier for this AgentsPartition
	int rank; // the rank of this partition
	int numElements;    // the number of agent elements in this AgentsPartition
	int Tbytes; // sizeof(agent)
	bool isloaded;
	bool loadable;
	int ghostWidth;
	dim3 dims[2]; // 0 is blockdim, 1 is threaddim
};
// end class

}// mass namespace
