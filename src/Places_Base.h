/**
 *  @file Places_Base.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef PLACES_BASE_H_
#define PLACES_BASE_H_

#include <string>
#include <vector>
#include "PlacesPartition.h"

namespace mass {

class Places_Base {

public:

	/**
	 *  Destructor
	 */
	virtual ~Places_Base();

	/**
	 *  Returns the number of dimensions in this places object. (I.e. 2D, 3D, etc...)
	 */
	virtual int getDimensions();

	/**
	 *  Returns the actual dimensions of the Places_Base matrix. The returned array will be getDimension() elements long.
	 */
	virtual int *size();

	/**
	 *  Returns the handle associated with this Places_Base object that was set at construction.
	 */
	virtual int getHandle();

	/**
	 *  Executes the given functionId on each Place element within this Places_Base.
	 *
	 *  @param functionId the function id passed to each Place element
	 */
	virtual void callAll(int functionId)= 0;

	/**
	 *  Executes the given functionId on each Place element within this Places_Base with
	 *  the provided argument.
	 *
	 *  @param functionId the function id passed to each Place element
	 *  @param argument the argument to be passed to each Place element
	 *  @param argSize the size in bytes of the argument
	 */
	virtual void callAll(int functionId, void *argument, int argSize)= 0;

	/**
	 *  Calls the function specified on all place elements by passing argument[i]
	 *  to place[i]'s function, and receives a value from it into (void *)[i] whose
	 *  element size is retSize bytes. In case of multi-dimensional Places_Base array,
	 *  'i' is considered as the index when the Places_Base array is flattened into row-major
	 *  order in a single dimension.
	 *
	 *  @param functionId the function id passed to each Place element
	 *  @param arguments the arguments to be passed to each Place element
	 *  @param argSize the size in bytes of each argument element
	 *  @param retSize the size in bytes of the return array element
	 */
	virtual void *callAll(int functionId, void *arguments[], int argSize,
			int retSize)= 0;

	//// TODO implement the call some functions
	// void callSome( int functionId, int dim, int index[] );
	// void callSome( int functionId, void *argument, int argSize, int dim, int index[] );
	// void *callSome( int functionId, void *arguments[], int argSize, int dim, int index[] );
	// exchangeSome( int handle, int functionId, Vector<int*> *destinations, int dim, int index[] );

	/**
	 *  This function causes all Place elements to call the function specified on all neighboring
	 *  place elements. The offsets to neighbors are defined in the destinations vector (a collection
	 *  of offsets from the caller to the callee place elements). The caller cell's outMessage is a
	 *  continuous set of arguments passed to the callee's method. The caller's inMessages[] stores
	 *  values returned from all callees. More specifically, inMessages[i] maintains a set of return
	 *  from the ith neighbor in destinations.
	 *  Example destinations vector:
	 *    vector<int*> destinations;
	 *    int north[2] = {0, 1}; destinations.push_back( north );
	 *    int east[2] = {1, 0}; destinations.push_back( east );
	 *    int south[2] = {0, -1}; destinations.push_back( south );
	 *    int west[2] = {-1, 0}; destinations.push_back( west );
	 */
	virtual void exchangeAll(int functionId,
			std::vector<int*> *destinations)= 0;

	/**
	 *  Exchanges the boundary places with the left and right neighboring nodes. 
	 */
	virtual void exchangeBoundary()= 0;

	// /**
	// *  Adds partitions to this Places_Base object.
	// */
	// virtual void addPartitions(PlacesPartition **part);

	// virtual PlacesPartition *getPartition(int rank);

	virtual int getNumPartitions() = 0;

protected:

	/**
	 *  Creates a Places_Base object. Only accessible from the dispatcher.
	 *
	 *  @param handle the unique identifier of this places collections
	 *  @param boundary_width the width of the border, in elements, to exchange between segments.
	 *  @param argument a continuous space of arguments used to initialize the places
	 *  @param argSize the size in bytes of the argument array
	 *  @param dimensions the number of dimensions in the places matrix (i.e. is it 1D, 2D, 3d?)
	 *  @param size the size of each dimension. This MUST be dimensions elements long.
	 */
	Places_Base(int handle, int boundary_width, void *argument, int argSize,
			int dimensions, int size[]);

	int handle;         // User-defined identifier for this Places_Base
	int numDims; // the number of dimensions for this Places_Base (i.e. 1D, 2D, 3D, etc...)
	int *dimensions; // dimensions of the grid in which these places are located. It must be numDims long
	int boundary_width; // the width of borders between sections
	void *argument;
	int argSize;
	Dispatcher *dispatcher; // the GPU dispatcher
};

} /* namespace mass */
#endif // PLACES_BASE_H_
