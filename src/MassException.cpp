/**
 *  @file MassException.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "MassException.h"

namespace mass {

MassException::MassException( std::string msg ) {
	message = msg;
}

MassException::~MassException( ) throw () {
}

} /* namespace mass */
