/**
 *  @file Places.h
 *  @author Nate Hart, Rob Jordan
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef PLACES_H_
#define PLACES_H_

#include <string>
#include <vector>
#include "Model.h"
#include "Place.h"

namespace mass {

template <typename T>
class Places {
	friend class Model;
public:
	
	// /* Constructs a Places object with the given handle and size.
	 // *
	 // * handle    : user-defined handle that identifies this Places object.
	 // * dimensions: three-dimensional size of the Places array
	 // */
	// Places( int handle, dim3 dimensions );
  
  /**
   *  
   */
  Places( int handle, std::string className, void *argument, int argSize, int dimensions, int size[] );

  Places( int handle, std::string className, void *argument, int argSize, int dimensions, ... );
  
  Places( int handle, std::string className, int boundary_width, void *argument, int argSize, int dimension, int size[] );

  Places( int handle, std::string className, int boundary_width, void *argument, int argSize, int dimension, ... );
  
	/**
	 *  Destructor
	 */
	~Places( );
      
  /**
   *  Returns the number of dimensions in this places object. (I.e. 2D, 3D, etc...)
   */
  int getDimensions();
  
  /**
   *  Returns the actual dimensions of the Places matrix. The returned array will be getDimension() elements long.
   */
  int *size()
  
	/**
	 *  Returns an array of the Place elements contained in this Places object. This is an expensive
	 *  operation since it requires memory transfer.
	 */
	T* getElements( );

	/**
	 *  Returns the handle associated with this Places object that was set at construction.
	 */
	int getHandle( );

	/**
	 *  Returns the 3D size of this Places object.
	 */
	dim3 getDimensions( );

  /**
	 *  Executes the given functionId on each Place element within this Places.
	 *
	 *  @param functionId the function id passed to each Place element
	 */
	void callAll( int functionId );
  
	/**
	 *  Executes the given functionId on each Place element within this Places with
	 *  the provided argument.
	 *
	 *  @param functionId the function id passed to each Place element
	 *  @param argument the argument to be passed to each Place element
	 *  @param argSize the size in bytes of the argument
	 */
	void callAll( int functionId, void *argument, int argSize );
  
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
  void *callAll( int functionId, void *arguments[], int argSize, int retSize );
  
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
  void exchangeAll( int handle, int functionId, std::vector<int*> *destinations );
  
  /**
   *  Exchanges the boundary places with the left and right neighboring nodes. 
   */
  void exchangeBoundary( );


private:

	int handle;         // User-defined identifier for this Places
  int numDims;        // the number of dimensions for this Places (i.e. 1D, 2D, 3D, etc...)
	int *dimensions;   // dimensions of the grid in which these places are located. It must be numDims long
	T *elements;        // host elements stored in row-major order
  int boundary_width; // the width of borders between sections
};

} /* namespace mass */
#endif // PLACES_H_
