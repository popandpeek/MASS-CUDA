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

class Dispatcher;

template<typename T>
class PlacesPartition {

public:

  PlacesPartition(int handle, int rank, int numElements, int ghostWidth, int n, int *dimensions)
          :hPtr(NULL),dPtr(NULL),handle(handle),rank(rank),numElements(numElements),isloaded(false){
    Tbytes = sizeof(T);
    setGhostWidth(ghostWidth, n, dimensions);
    setIdealDims();
  }

	/**
	 *  Destructor
	 */
	~PlacesPartition(){}
  
  /**
   *  Returns the number of place elements in this partition.
   */
  int size() {
    int numRanks = Mass::getPlaces(handle)->getNumPartitions();
    if(1 == numRanks){
      return numElements;
    }
    
    int retVal = numElements;
    if(0 ==rank || numRanks-1 == rank){
      // there is only one ghost width on an edge rank
      retVal -= ghostWidth;
    } else {
      retVal -= 2 * ghostWidth;
    }
    
    return retVal;
  }
  
  /**
   *  Returns the number of place elements and ghost elements.
   */
  int sizePlusGhosts() {
    return numElements;
  }
  
  /**
   *  Gets the rank of this partition.
   */
  int getRank(){
    return rank;
  }

	/**
	 *  Returns an array of the Place elements contained in this PlacesPartition object. This is an expensive
	 *  operation since it requires memory transfer.
	 */
	T *hostPtr() {
    T *retVal = hPtr;
    if(rank > 0){
      retVal += ghostWidth;
    }
    return retVal;
  }
  
  /**
   *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
   */
  T *hostPtrPlusGhosts() {
    return hPtr;
  }
  
  /**
   *  Returns the pointer to the GPU data. NULL if not on GPU.
   */
  T *devicePtr() {
    return dPtr;
  }
  
  void setDevicePtr(T *places)){
    dPtr = places;
  }

	/**
	 *  Returns the handle associated with this PlacesPartition object that was set at construction.
	 */
	int getHandle() {
    return handle;
  }
  
  /**
   *  Sets the start and number of places in this partition.
   */
  void setSection(T *start){
    hPtr = start;
  }
  
  void setQty(int qty){
    numElements = qty;
    setIdealDims();
  }
  
  bool isLoaded(){
    return isloaded;
  }
  
  void makeLoadable(){
    if ( !loadable ) {
      if ( dPtr != NULL ) {
        cudaFree( dPtr );
      }
      
      cudaMalloc( (void**) &dPtr, Tbytes * sizePlusGhosts( ) );
      loadable = true;
    }
  }
  
  T *load(cudaStream_t stream){
    makeLoadable();
      
    cudaMemcpyAsync( dPtr, hPtr, Tbytes * sizePlusGhosts( ), cudaMemcpyHostToDevice, stream );
    loaded = true;
  }

  
  bool retrieve(cudaStream_t stream, bool freeOnRetrieve){
    bool retreived = loaded;

    if ( loaded ) {
      cudaMemcpyAsync( hPtr, dPtr, Tbytes * sizePlusGhosts( ), cudaMemcpyDeviceToHost, stream );
      loaded = false;
    }

    if(freeOnRetreive){
      cudaFree(dPtr);
      loadable = false;
      dPtr = NULL;
    }

    return retreived;
  }
  
  int getGhostWidth(){
    return ghostWidth;
  }
  
  void setGhostWidth(int width, int n, int *dimensions){
    ghostWidth = width;
    
    // start at 1 because we never want to factor in x step
    for(int i = 1; i < n; ++i){
      ghostWidth += dimensions[i];
    }
  }
  
  void updateLeftGhost(T *ghost, cudaStream_t stream){
    if(rank > 0){
      if(isloaded){
        cudaMemcpyAsync( dPtr, ghost, Tbytes * ghostWidth, cudaMemcpyHostToDevice, stream );
      } else {
        memcpy(hPtr, ghost, Tbytes * ghostWidth);
      }
    }
  }
  
  void updateRightGhost(T *ghost, cudaStream_t stream){
    if(rank < PlacesPartition::numRanks-1){
      if(isloaded){
        cudaMemcpyAsync( dPtr + numElements, ghost, Tbytes * ghostWidth, cudaMemcpyHostToDevice, stream );
      } else {
        memcpy(hPtr + ghostWidth + numElements, ghost, Tbytes * ghostWidth);
      }
    }
  }
  
  T *getLeftBuffer(){
    if(isloaded){
      cudaMemcpy( hPtr, dPtr + ghostWidth, Tbytes * ghostWidth, cudaMemcpyDeviceToHost );
    } 
    
    return hPtr + ghostWidth;
  }
  
  T *getRightBuffer()(){
    if(isloaded){
      cudaMemcpy( hPtr, dPtr + numElements, Tbytes * ghostWidth, cudaMemcpyDeviceToHost );
    }
    return hPtr + numElements;
  }

  
  dim3 blockDim(){
    return dims[0];
  }
  
  dim3 threadDim(){
    return dims[1];
  }
  
  void setIdealDims(){
    int numBlocks = (numElements - 1) / THREADS_PER_BLOCK + 1;
    dim3 blockDim(numBlocks, 1, 1);
    
    int nThr = (numElements - 1) / numBlocks + 1;
    dim3 threadDim(nThr, 1, 1);
    
    dims[0] = blockDim;
    dims[1] = threadDim;
  }

  int getPlaceBytes(){
    return Tbytes;
  }

private:
  T *hPtr; // this starts at the left ghost, and extends to the end of the right ghost
  T *dPtr; // pointer to GPU data
  static int numRanks; // the overall number of ranks in this model
	int handle;         // User-defined identifier for this PlacesPartition
  int rank; // the rank of this partition
	int numElements;    // the number of place elements in this PlacesPartition
  int Tbytes; // sizeof(agent)
  bool isloaded;
  bool loadable;
  int ghostWidth;
  dim3 dims[2]; // 0 is blockdim, 1 is threaddim
};

} /* namespace mass */
#endif // PLACESPARTITION_H_
