/**
 * TestPlaceInstantiator.h
 *
 *  Author: Nate Hart
 *  Created on: Nov 12, 2014
 */

#ifndef TESTPLACEINSTANTIATOR_H_
#define TESTPLACEINSTANTIATOR_H_
#include "TestPlace.h"
namespace mass {

class TestPlaceInstantiator {
public:
	__device__ TestPlace *instantiate(void *arg);

};

} /* namespace mass */
#endif /* TESTPLACEINSTANTIATOR_H_ */
