
#pragma once

#include <exception>
#include <string>

namespace mass {

/**
 *  This is the standard exception to be thrown in the mass library.
 *  At this point we assume that all MassExceptions are fatal, i.e.
 *  the state cannot be recovered and the program using the library must
 *  end (implication: we don't attempt to clear memory before failing. 
 *  reasoning: if we do, it leads to infinite loop of expetions in case 
 *  memory is corrupted)
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
