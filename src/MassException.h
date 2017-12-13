/**
 *  @file MassException.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

#include <exception>
#include <string>

namespace mass {

/**
 *  This is the standard exception to be thrown in the mass library.
 */
class MassException: public std::exception {
public:
	/**
	 *  Constructor requires a std::string that will be returned to the caller
	 *  with the exception.
	 */
	MassException(std::string msg);
	virtual ~MassException() throw ();
	std::string message;
};

} /* namespace mass */
