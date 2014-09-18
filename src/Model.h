/**
 *  @file Model.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef MODEL_H_
#define MODEL_H_

#include <map>
#include <vector>
#include <string>
#include "Agents.h"
#include "Places.h"
#include "Slice.h"

namespace mass {

struct AgentsSlice{
  Agent *begin; // the first Agent in this slice
  int qty; // the number of agents in this slice
}

struct PlacesSlice{
  Place *begin; // the start of this slice of Places, and start of left buffer
  Place *leftGhost; // the start of the area belonging to the lower rank Places slice
  Place *rightGhost; // the start of the area belonging to the higher rank Places slice
  Place *rightBuffer; // the start of the area that needs to be sent to higher rank Places slice
  int qty; // the number of place elements in this slice
  int ghostWidth; // the number of elements to send to another rank when exchanging borders
}

class Model {
  std::map< int, Agents* > agentsMap;
  std::map< int, Places* > placesMap;
  std::map< int, Slice > slicesMap;

public:
  Model();
  ~Model();
  bool addAgents(Agents *agents);
  bool addPlaces(Places *places);
  Agents *getAgents( int handle );
  Places *getPlaces( int handle );
  Slice getSlice( int rank );
  int getNumSlices();
  void setNumSlices(int n);
  void endTurn(); // not yet implemented. Eventually will trigger "clean up" between turns

}; // end Model
} // end namespace
#endif // MODEL_H_