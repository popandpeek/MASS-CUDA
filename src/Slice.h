/**
 *  @file Slice.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef SLICE_H_
#define SLICE_H_

#include <map>
#include <string>
#include "Agent.h"
#include "Model.h"
#include "Place.h"


namespace mass {


struct AgentsSlice{
  Agent *begin; // the first Agent in this slice
  int qty; // the number of agents in this slice
}

struct PlacesSlice{
  Place *begin; // the start of this slice of Places
  Place *leftGhost; // the start of the area belonging to the lower rank Places slice
  Place *leftBuffer; // the start of the area to send to the lower rank Places slice
  Place *rightGhost; // the start of the area belonging to the higher rank Places slice
  Place *rightBuffer; // the start of the area to send to the higher rank Places slice
  
}

/**
 *  This class represents a GPU-sized chunk of the overall data model. This
 *  includes a portion of the Places, as well as all agents residing on those
 *  places.
 */
class Slice {
  int rank;
  bool isLoaded;
  std::vector<AgentsSlice> agents;
  std::vector<PlacesSlice> places;

public:
  Slice(int rank);
  ~Slice();

  /** load and unload functions. */
  void load(cudaStream_t stream);
  bool retreive(cudaStream_t stream, bool freeOnRetreive);
  bool isLoaded();

  int getRank();
}; // end Slice
} // end namespace
#endif // SLICE_H_