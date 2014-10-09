/*
 * TestPlace.cpp
 *
 *  Created on: Oct 8, 2014
 *      Author: natehart
 */

#include "TestPlace.h"
#include "src/Mass.h"

namespace mass {

TestPlace::TestPlace(void *arg) :
		Place(arg) {
	if (NULL != arg) {
		message = *((int*) arg);
	} else {
		message = -1;
	}
}

__host__ __device__ void *TestPlace::getMessage() {
	return &message;
}

/**
 * Returns the number of bytes necessary to store this agent implementation.
 * The most simple implementation is a single line of code:
 * return sizeof(*this);
 *
 * Because sizeof is respoved at compile-time, the user must implement this
 * function rather than inheriting it.
 *
 * @return an int >= 0;
 */MASS_FUNCTION unsigned TestPlace::placeSize() {
	return sizeof(*this);
}

MASS_FUNCTION void TestPlace::callMethod(int functionId, void *arg) {
	switch (functionId) {
	case SET_TO_ONE:
		setToOne();
		break;
	case SET_TO_ARG:
		setToArg((int*) arg);
		break;
	}
}

MASS_FUNCTION void TestPlace::setToOne() {
	message = 1;
}

MASS_FUNCTION void TestPlace::setToArg(int *arg) {
	if(NULL != arg){
		message = *arg;
	}
}

} /* namespace mass */
