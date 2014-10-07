/**
 *  @file Places.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef PLACES_H_
#define PLACES_H_

#include <string>
#include <vector>
#include "MassException.h"
//#include "PlacesPartition.h"


namespace mass {
// forward declarations
class Dispatcher;
class PlacesPartition;

class Places {

    friend class Dispatcher;
    friend class PlacesPartition;
public:

	/**
	 *  Destructor
	 */
    ~Places ( );

    /**
    *  Returns the number of dimensions in this places object. (I.e. 2D, 3D, etc...)
    */
    int getDimensions ( );

    /**
    *  Returns the actual dimensions of the Places_Base matrix. The returned array will be getDimension() elements long.
    */
    int *size ( );

    /**
    *  Returns the handle associated with this Places_Base object that was set at construction.
    */
    int getHandle ( );

	/**
	 *  Executes the given functionId on each Place element within this Places.
	 *
	 *  @param functionId the function id passed to each Place element
	 */
    void callAll ( int functionId );

	/**
	 *  Executes the given functionId on each Place element within this Places with
	 *  the provided argument.
	 *
	 *  @param functionId the function id passed to each Place element
	 *  @param argument the argument to be passed to each Place element
	 *  @param argSize the size in bytes of the argument
	 */
    void callAll ( int functionId, void *argument, int argSize );

	/**
	 *  Calls the function specified on all place elements by passing argument[i]
	 *  to place[i]'s function, and receives a value from it into (void *)[i] whose
	 *  element size is retSize bytes. In case of multi-dimensional Places array,
	 *  'i' is considered as the index when the Places array is flattened into row-major
	 *  order in a single dimension.
	 *
	 *  @param functionId the function id passed to each Place element
	 *  @param arguments the arguments to be passed to each Place element
	 *  @param argSize the size in bytes of each argument element
	 *  @param retSize the size in bytes of the return array element
	 */
    void *callAll ( int functionId, void *arguments[ ], int argSize,
                    int retSize );

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
    void exchangeAll ( int functionId, std::vector<int*> *destinations );

	/**
	 *  Exchanges the boundary places with the left and right neighboring nodes. 
	 */
    void exchangeBoundary ( );

	/**
	 *  Returns the Place elements contained in this Places object. This is an expensive
	 *  operation since it requires memory transfer. It is up to the caller to delete this array.
	 */
    Place** getElements ( );

    /**
     * Returns the row major index of the given coordinates. For instance, in an
     * int[3][5], the element (1,3) will have the row major index of 8.
     *
     * @param indices an series of ints ordered rowIdx,colIdx,ZIdx,etc...
     * that specify a single element in this places object. The number of
     * elements must be equal to the number of dimensions in this places object.
     * All indices must be non-negative.
     *
     * This is the inverse function of getIndexVector()
     *
     * @return an int representing the row-major index where this element is stored.
     */
    int getRowMajorIdx(int *indices);

    /**
     * This function will take a valid (in bounds) row-major index for this
     * places object and return a vector that contains the row, col, z, etc...
     * indices for the place element at the given row major index.
     *
     * This is the inverse function of getRowMajorIdx(...)
     *
     * @param rowMajorIdx the index of an element in a flattened mult-dimensional
     * array. Must be non-negative.
     *
     * @return a vector<int> with the multi-dimensional indices of the element.
     */
    std::vector<int> getIndexVector( int rowMajorIdx );

protected:

	/**
	 *  Creates a Places object. Only accessible from the dispatcher.
	 *
	 *  @param handle the unique identifier of this places collections
	 *  @param boundary_width the width of the border, in elements, to exchange between segments.
	 *  @param argument a continuous space of arguments used to initialize the places
	 *  @param argSize the size in bytes of the argument array
	 *  @param dimensions the number of dimensions in the places matrix (i.e. is it 1D, 2D, 3d?)
	 *  @param size the size of each dimension. This MUST be dimensions elements long.
	 */
    Places ( int handle, void *argument, int argSize,
             int dimensions, int size[ ], int boundary_width );


    void setTsize ( int size );

    int getTsize ( );

    int getNumPartitions ( );

    /**
    *  Adds partitions to this Places object.
    */
    void addPartitions ( std::vector<PlacesPartition*> parts );

    /**
    *  Gets a partition from this Places object.
    */
    PlacesPartition *getPartition ( int rank );


    int handle;         // User-defined identifier for this Places_Base
    int numDims; // the number of dimensions for this Places_Base (i.e. 1D, 2D, 3D, etc...)
    int *dimensions; // dimensions of the grid in which these places are located. It must be numDims long
    int boundary_width; // the width of borders between sections
    void *argument;
    int argSize;
    Dispatcher *dispatcher; // the GPU dispatcher
    unsigned numElements;
    Place **elemPtrs;
    unsigned Tsize;
	void *elements;
	std::map<int, PlacesPartition*> partitions;
};

} /* namespace mass */
#endif // PLACES_H_
