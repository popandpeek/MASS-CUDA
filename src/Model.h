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


class Model {
  std::map< int, Agent* > agentsMap;
  std::map< int, Place* > placesMap;
  std::map< int, Slice > slices;

public:
  Model();
  ~Model();
  bool addAgents(Agents *agents);
  bool addPlaces(Places *places);
  int getNumSlices();
  void setNumSlices(int n); // not yet implemented
  void endTurn(); // not yet implemented

  /************************************************************
   *  ITERATOR FUNCTIONS
  ************************************************************/
  bool hasNextSlice(); // not yet implemented 
  Slice *nextSlice(); // not yet implemented

}; // end Model
} // end namespace
#endif // MODEL_H_