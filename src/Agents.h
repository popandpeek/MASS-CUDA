/**
 *  @file Agents.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef AGENTS_H_
#define AGENTS_H_

#include <string>
#include <vector>
#include "Agent.h"
#include "Model.h"
#include "Places.h"

namespace mass{

template <typename T>
class Agents {
  friend class Model;

public:

  Agents( int handle, std::string className, void *argument, int argSize, Places *places, int initPopulation );

  ~Agent();

  int getHandle( );

  int nAgents( );
  
  void callAll( int functionId );

  void callAll( int functionId, void *argument, int argSize);
  
  void *callAll( int functionId, void *arguments[], int argSize, int retSize )

  void manageAll( );

private:

  Places *places; /**< The places used in this simulation. */

  T* agents; /**< The agents elements.*/

  int handle; /**< Identifies the type of agent this is.*/

  int numAgents; /**< Running count of living agents in system.*/
  
  int newChildren; /**< Added to numAgents and reset to 0 each manageAll*/

  int sequenceNum; /*!< The number of agents created overall. Used for agentId creation. */
} // end class

} // mass namespace

#endif // AGENTS_H_
