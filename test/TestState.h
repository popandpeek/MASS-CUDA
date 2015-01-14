/**
 * TestState.h
 *
 *  Author: Nate Hart
 *  Created on: Nov 18, 2014
 */

#ifndef TESTSTATE_H_
#define TESTSTATE_H_

#include "../src/PlaceState.h"

namespace mass {


class TestState
		: public PlaceState {

public:
	int message;
};

} // end namespace

#endif /* TESTSTATE_H_ */
