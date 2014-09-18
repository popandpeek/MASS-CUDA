/**
 *  @file Slice.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "cudaUtil.h"
#include "Slice.h"

namespace mass {

  Slice::Slice(int rank){
    this->rank = rank;
    this->isloaded = false;
  }
  Slice::~Slice(){
    agents.empty();
    places.empty();
  }
  
  bool Slice::addAgentsSlice(AgentSlice slice){
    agents[slice.handle] = slice;
  }
  
  AgentsSlice Slice::getAgents( int handle ){
    std::map<int, AgentsSlice>::iterator it = agents.find(handle);
    if(it == agents.end()){
      throw MassException("Agents handle not found.");
    }
    return it->second;
  }
  
  int Slice::getNumAgents(){
    return agents.size();
  }
  
  bool Slice::addPlacesSlice(PlacesSlice slice){
    places[slice.handle] = slice;
  }
  
  PlacesSlice Slice::getPlaces( int handle ){
    std::map<int, PlacesSlice>::iterator it = places.find(handle);
    if(it == places.end()){
      throw MassException("Places handle not found.");
    }
    return it->second;
  }
  
  int Slice::getNumPlaces(){
    return places.size();
  }

  /** load and unload functions. */
  void Slice::load(cudaStream_t stream){
    std::map<int, AgentsSlice>::iterator it = agents.begin();
    while(it != agents.end()){
      AgentsSlice slice = it->second;
      size_t count = slice.qty * sizeof(slice.begin[0]);
      CHECK( cudaMalloc( (void**) &slice.d_begin, count) );
      CHECK( cudaMemcpyAsync( slice.d_begin, slice.begin, count, cudaMemcpyHostToDevice, stream ) );
    }
    
    std::map<int, PlacesSlice>::iterator it = places.begin();
    while(it != places.end()){
      PlacesSlice slice = it->second;
      size_t count = slice.qty * sizeof(slice.begin[0]);
      CHECK( cudaMalloc( (void**) &slice.d_begin, count) );
      CHECK( cudaMemcpyAsync( slice.d_begin, slice.begin, count, cudaMemcpyHostToDevice, stream ) );
    }
    isloaded = true;
  }
  
  
  bool Slice::retreive(cudaStream_t stream, bool freeOnRetreive){
    std::map<int, AgentsSlice>::iterator it = agents.begin();
    while(it != agents.end()){
      AgentsSlice slice = it->second;
      size_t count = slice.qty * sizeof(slice.begin[0]);
      CHECK( cudaMemcpyAsync( slice.begin, slice.d_begin, count, cudaMemcpyDeviceToHost, stream ) );
      if(freeOnRetreive){
        CHECK( cudaFree( slice.d_begin ) );
      }
    }
    
    std::map<int, PlacesSlice>::iterator it = places.begin();
    while(it != places.end()){
      PlacesSlice slice = it->second;
      size_t count = slice.qty * sizeof(slice.begin[0]);
      CHECK( cudaMemcpyAsync( slice.begin, slice.d_begin, count, cudaMemcpyDeviceToHost, stream ) );
      if(freeOnRetreive){
        CHECK( cudaFree( slice.d_begin ) );
      }
    }
    isloaded = !freeOnRetreive; // if free == true, then isloaded = false
  }
  
  bool Slice::isLoaded(){
    return isloaded;
  }

  int Slice::getRank(){
    return rank;
  }
  
}; // end Slice
} // end namespace
