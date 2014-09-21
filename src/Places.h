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
#include "Dispatcher.h"
#include "Places_Base.h"

namespace mass {

template<typename T>
class Places : public Places_Base {

friend class Dispatcher;
public:

	/**
	 *  Destructor
	 */
	virtual ~Places(){
    if(NULL != elements){
      delete[] elements;
    }
    partitions.empty();
  }
  /**
	 *  Executes the given functionId on each Place element within this Places_Base.
	 *
	 *  @param functionId the function id passed to each Place element
	 */
	virtual void callAll(int functionId) {
    callAll(functionId, NULL, 0);
  }


	/**
	 *  Executes the given functionId on each Place element within this Places_Base with
	 *  the provided argument.
	 *
	 *  @param functionId the function id passed to each Place element
	 *  @param argument the argument to be passed to each Place element
	 *  @param argSize the size in bytes of the argument
	 */
	virtual void callAll(int functionId, void *argument, int argSize) {
    dispatcher->callAllPlaces<T>(this, functionId, argument, argSize);
  }

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
	virtual void *callAll(int functionId, void *arguments[], int argSize, int retSize) {
    return dispatcher->callAllPlaces<T>(this, functionId, arguments, argSize, retSize);
  }

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
	virtual void exchangeAll(int functionId, std::vector<int*> *destinations) {
    dispatcher->exchangeAllPlaces<T>(this, functionId, destinations);
  }

	/**
	 *  Exchanges the boundary places with the left and right neighboring nodes. 
	 */
	virtual void exchangeBoundary() {
    dispatcher->exchangeBoundaryPlaces<T>(this);
  }

	/**
	 *  Returns the Place elements contained in this Places object. This is an expensive
	 *  operation since it requires memory transfer. It is up to the caller to delete this array.
	 */
	virtual T* getElements(){
    dispatcher->refreshPlaces<T>(this);
    return elements;
  }
  
  vitrual int getNumPartitions(){
    return partitions.size();
  }
  
  /**
   *  Adds partitions to this Places object.
   */
  virtual void addPartitions(std::vector<PlacesPartition<T>*> parts){
  
  // make sure add is valid
  if(NULL == elements){
  
    // calculate number of elements in this collection
    int numElem = 1;
    for(int i = 0; i < dimensions; ++i){
      numElem *= size[i];
    }

    elements = new T[numElem];
    T *dst = elements;
    int numRanks = parts.size();
    
    // we have only one rank, give it all Places
    if(0 == i && 1 == numRanks){
      part->setSection(dst);
      part->setQty(numElem);
    } else {
      // we have multiple ranks. first numRanks-1 get full slice, nth gets remainder
      int partitionSize = numElem / numRanks;
      int remainder = numElem % partitionSize;
      
      // for each part
      for(int i = 0; i < numRanks; ++i){
        PlacesPartition<T> *part = parts[i];
        partitions[part->getRank()] = part;
        part->setSection(dst);
        
        if(i < numRanks-1){
          part->setQty(partitionSize);
        } else {
          part->setQty(remainder);
        }
        
        if(i > 0){
          dst += partitionSize; // move 
        } else {
          // rank 0 doesn't have a first ghost rank, so it needs to move less
          dst += (partitionSize - part->getGhostWidth());
        } 
      }
    }
  }
}
  
  /**
   *  Gets a partition from this Places object.
   */
  PlacesPartition<T> *getPartition(int rank){
    if(rank < 0 || rank >= partitions.size()){
      throw MassException("Out of bounds rank specified in Places::getPartition()");
    }
    return partitions[rank];
  }

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
	Places(int handle, int boundary_width, void *argument, int argSize,
			int dimensions, int size[]):Places_Base(handle, boundary_width, argument, argSize, dimensions, size){
    
    elements = NULL;
  }

  T *elements;
  std::map<int,PlacesPartition<T>*> partitions;
};

} /* namespace mass */
#endif // PLACES_H_
