/*
 * TestPlace.h
 *
 *  Created on: Oct 8, 2014
 *      Author: natehart
 */

#ifndef TESTPLACE_H_
#define TESTPLACE_H_

#include "src/Place.h"

namespace mass {



class TestPlace: public Place {

	int message;

public:
	/**
	 * The function IDs
	 */
	enum{
		SET_TO_ONE,//!< SET_TO_ONE
		SET_TO_ARG //!< SET_TO_ARG
	};

	TestPlace(void *argument);

	/**
	 *  Gets a pointer to this place's out message.
	 */
	__host__ __device__ virtual void *getMessage();

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
	MASS_FUNCTION virtual unsigned placeSize();


	/**
	 * Calls a function specified within the class enum.
	 */
	MASS_FUNCTION virtual void callMethod( int functionId, void *arg);

	/**
	 * Sets message to 1. Specified using SET_TO_ONE enum.
	 */
	MASS_FUNCTION void setToOne();

	/**
	 * Sets message to value of arg, if arg is not null. Specified using SET_TO_ARG enum.
	 *
	 * @param arg a non-NULL integer pointer.
	 */
	MASS_FUNCTION void setToArg(int *arg);
};

} /* namespace mass */
#endif /* TESTPLACE_H_ */
