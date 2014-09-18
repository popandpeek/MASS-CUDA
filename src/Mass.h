/**
 *  @file Mass.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef MASS_H_
#define MASS_H_

#include <iostream>
#include <stddef.h>
#include <map>
#include "Dispatcher.h"
#include "Model.h"
#include "Places.h"

#define WARP_SIZE 32    // threads per warp
#define BLOCK_SIZE 512  // max threads per block

namespace mass {

class Mass {
public:

	/**
	 *  Initializes the MASS environment. Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 *  @param ngpu the number of GPUs to use
	 */
	static void init( String[] args, int ngpu );
  
  /**
	 *  Initializes the MASS environment using all available GPU resources. 
	 *  Must be called prior to all other MASS methods.
	 *  @param args what do these do?
	 */
	static void init( String[] args );

	/**
	 *  Shuts down the MASS environment, releasing all resources.
	 */
	static void finish( );
  
  /**
   *  Gets the places object for this handle.
   *  @param handle an int that corresponds to a places object.
   *  @return NULL if not found.
   */
  static Places *getPlaces( int handle );
  
  /**
   *  Gets the agents object for this handle.
   *  @param handle an int that corresponds to an agents object.
   *  @return NULL if not found.
   */
  static Agents *getAgents( int handle );
  
  /**
   *  Returns an instance of mass to the caller.
   */
  static Mass &getInstance(){
    static Mass instance; // Guaranteed to be destroyed. Instantiated on first use.
    return instance;
  }

private:
	// this is a singleton
	Mass( ){};
  Mass(Mass const&); // Don't Implement. No copies are allowed.
  void operator=(Mass const&); // Don't implement, prevent assignment.
  
  Model model; /**< The data model for this simulation. */
  Dispatcher dispatcher;/**< The object that handles communication with the GPU(s). */
};

} /* namespace mass */
#endif // MASS_H_
