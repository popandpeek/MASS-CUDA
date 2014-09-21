/**
 *  @file Places_Base.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "Mass.h"
#include "Places_Base.h"
#include "Dispatcher.h"

namespace mass {

Places_Base::~Places_Base() {
}

int Places_Base::getDimensions() {
	return numDims;
}

int *Places_Base::size() {
	return dimensions;
}

int Places_Base::getHandle() {
	return handle;
}

Places_Base::Places_Base(int handle, int boundary_width, void *argument, int argSize,
		int dimensions, int size[]) {
	this->handle = handle;
	this->numDims = dimensions;
	this->dimensions = size;
	this->boundary_width = boundary_width;
  this->argument = argument;
  this->argSize = argSize;
}

} /* namespace mass */
