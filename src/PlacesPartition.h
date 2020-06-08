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

#include "Place.h"

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
	 */
	PlacesPartition(Place** hp,  int handle, int rank, int numElements, int ghostWidth,
			int n, int *dimensions);

	/**
	 *  Destructor
	 */
	~PlacesPartition();

	/**
	 *  Gets the rank of this partition.
	 */
	int getRank();

	/**
	 *  Returns the number of place elements in this partition.
	 */
	int size();

	/**
	 *  Returns the number of place elements plus ghost elements.
	 */
	//int sizeWithGhosts();

	/**
	 *  Returns the handle associated with this PlacesPartition object that was set at construction.
	 */
	int getHandle();

	/**
	 *  Sets the start and number of places in this partition.
	 */
	//void setSection(Place **start);

	/**
	 * Returns the number of elements in the ghost width.
	 * @return
	 */
	//int getGhostWidth();

	/**
	 * Sets the ghost width to width X elements. Calculates the actual number
	 * of elements using n and dimensions, then sets pointers accordingly.
	 * @param width the number of rows in the X direction to exchange between turns.
	 * @param n the number of dimensions
	 * @param dimensions the size of each n dimensions
	 */
	//void setGhostWidth(int width, int n, int *dimensions);

	/**************************************************************************
	 * A block of a places element is laid out thusly:
	 *
	 * 1. Left ghost
	 * 2. Left Buffer
	 * 3. Right Buffer
	 * 4. Right Ghost
	 *
	 *       |-------This rank's data-------|
	 *   ----------------------------------------
	 *   | 1 |/2/|//////////////////////|/3/| 4 |
	 *   |   |///|//////////////////////|///|   |
	 *   |   |///|//////////////////////|///|   |
	 *   |   |///|//////////////////////|///|   |
	 *   |   |///|//////////////////////|///|   |
	 *   |   |///|//////////////////////|///|   |
	 *   |   |///|//////////////////////|///|   |
	 *   ----------------------------------------
	 *
	 * The block of memory on the GPU spans 1 - 4, but a rank should only copy
	 * out data between 2 & 3. Ghost space is to allow a rank to "reach across"
	 * the rank borders for shared computation. Rank 1's left buffer is rank 0's
	 * right ghost space.
	 */

	/**
	 * Returns a pointer to the left buffer.
	 */
	//Place *getLeftBuffer();

	/**
	 * Returns a pointer to the right buffer.
	 */
	//Place *getRightBuffer();

	/**
	 * Returns a pointer to the start of the left ghost space.
	 */
	//Place *getLeftGhost();

	/**
	 * Returns a pointer to the start of the right ghost space.
	 */
	//Place *getRightGhost();

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
	 * Gets the number of bytes for a single state element in the Places instance
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

	Place **hPtr; // this starts at the left ghost
	int handle;         // User-defined identifier for this PlacesPartition
	int rank; // the rank of this partition
	int numElements; // the number of place elements in this PlacesPartition from left ghost to right ghost
	int Tsize; // TODO remove
	//int ghostWidth;
	dim3 dims[2]; // 0 is blockdim, 1 is threaddim
};

} /* namespace mass */
#endif // PLACESPARTITION_H_
