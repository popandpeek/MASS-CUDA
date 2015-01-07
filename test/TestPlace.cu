/*
 * TestPlace.cpp
 *
 *  Created on: Oct 8, 2014
 *      Author: natehart
 */


#include "../src/Mass.h"

#include "TestPlace.h"
#include "TestState.h"

namespace mass {

MASS_FUNCTION TestPlace::TestPlace(void *arg):Place(arg){
	if (NULL != arg) {
		((TestState*) state)->message = *((int*) arg);
	} else {
		((TestState*) state)->message = -1;
	}
}

MASS_FUNCTION void *TestPlace::getMessage() {
	return &((TestState*) state)->message;
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
 */
MASS_FUNCTION int TestPlace::placeSize() {
	return sizeof(TestPlace);
}

MASS_FUNCTION void TestPlace::callMethod(int functionId, void *arg) {
	((TestState*) state)->message = -5;
	switch (functionId) {
	case SET_TO_ONE:
		setToOne();
		break;
	case SET_TO_ARG:
		setToArg((int*) arg);
		break;
	default:
		((TestState*) state)->message = -127;
	}
}

MASS_FUNCTION void TestPlace::setToOne() {
	((TestState*) state)->message = 1;
}

MASS_FUNCTION void TestPlace::setToArg(int *arg) {
	if(NULL != arg){
		((TestState*) state)->message = *arg;
	} else {
		((TestState*) state)->message = -10; // null argument
	}
}

//extern "C"
//MObject *instantiate( void *argument ){
//  return new TestPlace(argument);
//}
//
//extern "C"
//void destroy( MObject *obj ){
//  if(NULL != obj){
//    delete obj;
//  }
//}

} /* namespace mass */
