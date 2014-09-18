/**
 *  @file Model.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Model.h"
#include "math.h" // ciel

namespace mass {
// std::map< int, Agents* > agentsMap;
// std::map< int, Places* > placesMap;
// std::map< int, Slice > slicesMap;

void addAgentsToSlices(Agents *agents){
  Agent* elements = agents->agents; // friend access
  int size = (double) agents->nAgents();
  int numSlices = this->getNumSlices();
  int sliceSize = (int) ceil( ((double) size) / numSlices);
  int remainder = size - sliceSize * (numSlices-1);
  
  int rank=0; // first n-1 ranks get a full slice
  for( ; rank < size - 1; ++rank){
    Slice slice = slicesMap.find(rank)->second;
    AgentSlice a;
    a.begin = elements + rank * sliceSize;
    a.qty = sliceSize;
    slice.addAgentSlice(a);
  }
  
  if(0 != rank){ // special case needed for single rank
    ++rank;
  }
  
  // last rank gets remainder
  Slice slice = slicesMap.find(rank)->second;
  AgentSlice a;
  a.begin = elements + rank * sliceSize;
  a.qty = remainder;
  slice.addAgentSlice(a);
}

// converts a given index into row major index provided the places dimensions
int toRowMajorIdx(int n, int *dims, int *idx){
  int row_major_index = 0;
  for(int i = 0; i < n; ++i){
    int indexValue = idx[i];
    int stride = 1; // the number of elements to jump
    // multiply all remaining dimensions
    for(int j = i+1; j < n; ++j){
      stride *= dims[j];
    }
    row_major_index += indexValue * stride;
  }
}

int getGhostWidth(Places *places){
  int boundary_width = places->bboundary_width; // friend access
  int stride = 0;
  // TODO figure out how to calculate this
  
  return stride;
}

void addPlacesToSlices(Places *places){
  Place *elements = places->elements; // friend access
  int n = places->numDims;
  int *dimensions = places->dimensions;
  int size =1;
  for(int i = 0; i < n; ++i){
    size *= dimensions[i];
  }
  
  int numSlices = this->getNumSlices();
  int sliceSize = (int) ceil( ((double) size) / numSlices);
  int remainder = size - sliceSize * (numSlices-1);
  
  if(1 == numSlices){ // special case for a single slice
    Slice slice = slicesMap.find(0)->second;
    PlacesSlice p;
    p.begin = elements;
    p.leftGhost = NULL;
    p.rightGhost = NULL;
    p.rightBuffer = NULL;
    p.ghostWidth = 0;
    slice.addPlacesSlice(p);
  } else {
    int ghostWidth = getGhostWidth(places);
  
    int rank=0; // first n-1 ranks get a full slice
    for( ; rank < size - 1; ++rank){
      Slice slice = slicesMap.find(rank)->second;
      PlacesSlice p;
      p.begin = elements + rank * sliceSize;
      if(0 == rank){
        p.leftGhost = NULL;
      } else {
        p.leftGhost = p.begin - ghostWidth; 
      }
      p.rightGhost = p.begin + sliceSize + ghostWidth; 
      p.rightBuffer = ghostWidth; 
      p.ghostWidth = ghostWidth;
      slice.addPlacesSlice(p);
    }
    
    // last rank gets remainder
    ++rank;
    Slice slice = slicesMap.find(rank)->second;
    PlacesSlice p;
    p.begin = elements + rank * sliceSize;
    p.leftGhost = ghostWidth; 
    p.rightGhost = NULL; 
    p.rightBuffer = NULL;
    p.ghostWidth = ghostWidth;
    slice.addPlaceSlice(p);
  }
}

Model::Model(){
  // for the time being, there is only ever one slice
  int rank = 0;
  Slice slice(0);
  slicesMap.insert( std::pair<int,Slice>(rank, slice) );
}

Model::~Model(){
  agentsMap.empty();
  placesMap.empty();
  slicesMap.empty();
}

bool Model::addAgents(Agents *agents){
  bool isNew = true;
  int handle = agents->getHandle();
  
  std::map<int,Agents*>::iterator it = agentsMap.begin();
  while(it != agentsMap.end() && isNew){
    if(it->first == handle){ // this collection is already in model
      isNew = false;
    }
  }
  
  if(isNew){
    agentsMap.insert( std::pair<int,Agents*>(handle, agents) );
    addAgentsToSlices(agents);  // update the data model
  }
  return isNew;
}

bool Model::addPlaces(Places *places){
  bool isNew = true;
  int handle = places->getHandle();
  
  std::map<int,Places*>::iterator it = placesMap.begin();
  while(it != placesMap.end() && isNew){
    if(it->first == handle){ // this collection is already in model
      isNew = false;
    }
  }
  
  if(isNew){
    placesMap.insert( std::pair<int,Places*>(handle, places) );
    addPlacesToSlices(places); // update the data model
  }
  return isNew;
}

Agents *Model::getAgents( int handle ){
  Agent *agents = NULL;
  std::map<int,Agents*>::iterator it = agentsMap.find( handle );
  if(it != agentsMap.end()){
    agents = it->second;
  }
  return agents;
}

Places *Model::getPlaces( int handle ){
  Place *places = NULL;
  std::map<int,Places*>::iterator it = placesMap.find( handle );
  if(it != placesMap.end()){
    places = it->second;
  }
  return places;
}

int Model::getNumSlices(){
  return slicesMap.size();
}

void Model::setNumSlices(int n){} // not yet implemented
void Model::endTurn(){} // not yet implemented

/************************************************************
 *  ITERATOR FUNCTIONS
************************************************************/
bool Model::hasNextSlice(){return false;} // not yet implemented 
Slice *Model::nextSlice(){return NULL;} // not yet implemented


} // end namespace
