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

	/**
	 * Constructor.
	 * @param handle the handle of the places instance this partition belongs to
	 * @param rank the rank of this place in range [0,n)
	 * @param numElements the number of elements in this partition, not including ghost width
	 * @param ghostWidth the number of x elements to exchange in ghost space
	 * @param n the number of dimensions in this places instance
	 * @param dimensions the size of the n dimensions
	 * @param Tsize the number of bytes in a place instance
	 */
	PlacesPartition(int handle, int rank, int numElements, int ghostWidth,
			int n, int *dimensions, int Tsize);

	/**
	 *  Destructor
	 */
	~PlacesPartition();

	/**
	 *  Returns the number of place elements in this partition.
	 */
	int size();

	/**
	 *  Returns the number of place elements and ghost elements.
	 */
	int sizePlusGhosts();

	/**
	 *  Gets the rank of this partition.
	 */
	int getRank();

	/**
	 *  Returns an array of the Place elements contained in this PlacesPartition object. This is an expensive
	 *  operation since it requires memory transfer.
	 */
	void *hostPtr();

	/**
	 *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
	 */
	void *hostPtrPlusGhosts();

	/**
	 *  Returns the pointer to the GPU data. NULL if not on GPU.
	 */
	void *devicePtr();

	void setDevicePtr(void *places);

	/**
	 *  Returns the handle associated with this PlacesPartition object that was set at construction.
	 */
	int getHandle();

	/**
	 *  Sets the start and number of places in this partition.
	 */
	void setSection(void *start);

	/**
	 * Queries to see if this partition is loaded on a device.
	 * @return true if it is loaded
	 */
	bool isLoaded();

	/**
	 * Set the loaded status of this partition.
	 * @param loaded
	 */
	void setLoaded(bool loaded);

	/**
	 * Returns the number of elements in the ghost width.
	 * @return
	 */
	int getGhostWidth();

	/**
	 * Sets the ghost width to width X elements. Calculates the actual number
	 * of elements using n and dimensions, then sets pointers accordingly.
	 * @param width the number of rows in the X direction to exchange between turns.
	 * @param n the number of dimensions
	 * @param dimensions the size of each n dimensions
	 */
	void setGhostWidth(int width, int n, int *dimensions);

	/**
	 * Returns a pointer to the left buffer.
	 */
	void *getLeftBuffer();

	/**
	 * Returns a pointer to the right buffer.
	 */
	void *getRightBuffer();

	/**
	 * Returns a pointer to the start of the left ghost space.
	 */
	void *getLeftGhost();

	/**
	 * Returns a pointer to the start of the right ghost space.
	 */
	void *getRightGhost();

	/**
	 * Returns the ideal block dimension for this partition. Used for launching
	 * kernel functions on this partition's data.
	 *
	 * @return
	 */
	dim3 blockDim();

	/**
	 * Returns the ideal thread dimension for this partition. Used for launching
	 * kernel functions on this partition's data.
	 *
	 * @return
	 */
	dim3 threadDim();

	/**
	 * Gets the number of bytes for a single place element in the Places instance
	 * this partition belongs to.
	 *
	 * @return an int >= 0
	 */
	int getPlaceBytes();

private:

	/**
	 * Refreshes the ideal dimensions for kernel launches. This should be called
	 * only when the partition is created or ghost width changes.
	 */
	void setIdealDims();

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
