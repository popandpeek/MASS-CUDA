/**
 *  @file Mass.h
 *  @author Nate Hart, Rob Jordan
 *  Some content adapted from the Cuda Toolkit Documentation: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "mass.h"

namespace mass {
  // Model model; /**< The data model for this simulation. */
  // Dispatcher dispatcher;/**< The object that handles communication with the GPU(s). */
  
Mass::Mass( ){
  // do nothing
}

void Mass::init( String[] args, int ngpu ){
// move this section to the Dispatcher
  // // adapted from the Cuda Toolkit Documentation: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  // int deviceCount;
  // cudaGetDeviceCount(&deviceCount);
  // for (int device = 0; device < deviceCount; ++device) {
      // cudaDeviceProp deviceProp;
      // cudaGetDeviceProperties(&deviceProp, device);
      // printf("Device %d has compute capability %d.%d.\n",
             // device, deviceProp.major, deviceProp.minor);
  // }
  dispatcher.init(ngpu, &model);
}

void Mass::init( String[] args ){
  dispatcher.init(0, &model);
}

void Mass::finish( ){
  // nothing to do here. model and dispatcher destructors will take care of everything.
}

Places *Mass::getPlaces( int handle ){
  return model.getPlaces( handle );
}

Agents *Mass::getAgents( int handle ){
  return model.getAgents( handle );
}

} /* namespace mass */
