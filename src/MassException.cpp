
#include "MassException.h"

namespace mass {

MassException::MassException(std::string msg) {
	message = msg;
}

MassException::~MassException() throw () {
}

} /* namespace mass */
