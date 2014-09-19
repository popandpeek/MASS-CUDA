/**
 *  @file Dispatcher.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef DISPATCHER_H_
#define DISPATCHER_H_
#include<cuda_runtime.h>
#include "Command.h"
#include "Model.h"
#include "Slice.h"

namespace mass {

class Dispatcher {

public:

  /**
   *  Is the Dispatcher constructor. 
   *  The Dispatcher must be initialized prior to use.
   */
  Dispatcher();

  /**
   *  Is the Dispatcher initializer. 
   *  the number of GPUs is passed to the initializer. The Dispatcher
   *  then locates the GPUs, sets up communication links, and prepares to begin
   *  dispatching data to and from the GPU.
   *  
   *  @param ngpu the number of GPUs to use in this simulation. 0 if all GPU resources are to be used.
   *  @param models the data model for this simulation
   */
  init(int ngpu, Model *model);


  ~Dispatcher();
  
  /**
   *  Implementation of the command design pattern. Takes a command object and
   *  returns whatever value comes back from the command. 
   *
   *  @param command a command object to execute.
   */
  std::vector<void*> executeCommand( Command *command );
  
  /**
   *  Creates a Places object.
   *  
   *  @param handle the unique identifier of this places collections
   *  @param argument a continuous space of arguments used to initialize the places
   *  @param argSize the size in bytes of the argument array
   *  @param dimensions the number of dimensions in the places matrix (i.e. is it 1D, 2D, 3d?)
   *  @param size the size of each dimension. This MUST be dimensions elements long.
   */
  template<typename T>
  Places( int handle, void *argument, int argSize, int dimensions, int size[] ){
    
  }

private:
	int ngpu;                   // number of GPUs in use
	int* devices;               // array of GPU device ids
	cudaStream_t* streams;      // cuda execution streams, two per device
	cudaEvent_t* events; // cuda events to synchronize execution streams, one per device
	Model *model; // the data model for this simulation
};
// end class
}// namespace mass

#endif
