/**
 *  @file Command.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef COMMAND_H_
#define COMMAND_H_

namespace mass{
/**
 *  This class is the command interface. It allows command objects to be created that
 *  store all functions and parameters to be used in executing a kernel function.
 *  Create and implement these functions as necessary.
 */
class Command{

public:
  virtual void *execute() = 0;
  virtual bool forcesExcution() = 0;
  
}; // end class

} // end namespace

#endif // COMMAND_H_