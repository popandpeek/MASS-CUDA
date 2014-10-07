/**
 *  @file PlacesPartition.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef PLACESPARTITION_H_
#define PLACESPARTITION_H_

#define THREADS_PER_BLOCK 512

#include <cuda_runtime.h>
#include <string>
#include <vector>
//#include "Mass.h"
//#include "Place.h"
//#include "Places.h"

namespace mass {

class PlacesPartition {
    friend class Places;

public:

    PlacesPartition ( int handle, int rank, int numElements, int ghostWidth,
                      int n, int *dimensions );
	/**
	 *  Destructor
	 */
    ~PlacesPartition ( );

	/**
	 *  Returns the number of place elements in this partition.
	 */
    int size ( );

	/**
	 *  Returns the number of place elements and ghost elements.
	 */
    int sizePlusGhosts ( );

	/**
	 *  Gets the rank of this partition.
	 */
    int getRank ( );

	/**
	 *  Returns an array of the Place elements contained in this PlacesPartition object. This is an expensive
	 *  operation since it requires memory transfer.
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

    void setDevicePtr ( void *places );

	/**
	 *  Returns the handle associated with this PlacesPartition object that was set at construction.
	 */
    int getHandle ( );

	/**
	 *  Sets the start and number of places in this partition.
	 */
    void setSection ( void *start );

    void setQty ( int qty );

    bool isLoaded ( );

    void setLoaded ( bool loaded );

    void makeLoadable ( );

    void load ( cudaStream_t stream );

    bool retrieve ( cudaStream_t stream, bool freeOnRetrieve );

    int getGhostWidth ( );

    void setGhostWidth ( int width, int n, int *dimensions );

	void updateLeftGhost(void *ghost, cudaStream_t stream);

    void updateRightGhost ( void *ghost, cudaStream_t stream );

    void *getLeftBuffer ( );

    void *getRightBuffer ( );

    dim3 blockDim ( );

    dim3 threadDim ( );

    void setIdealDims ( );

    int getPlaceBytes ( );

private:
	void *hPtr; // this starts at the left ghost, and extends to the end of the right ghost
	void *dPtr; // pointer to GPU data
	int handle;         // User-defined identifier for this PlacesPartition
	int rank; // the rank of this partition
	int numElements;    // the number of place elements in this PlacesPartition
	int Tsize; // sizeof(agent)
	bool isloaded;
	bool loadable;
	int ghostWidth;
	dim3 dims[2]; // 0 is blockdim, 1 is threaddim
};

} /* namespace mass */
#endif // PLACESPARTITION_H_
