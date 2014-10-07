/**
 *  @file MObject.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once

namespace mass {

/**
 * This is the base object for all objects in the MASS library. Much of the base
 * functionality of objects is defined here and inherited for ease of use.
 */
class MObject {
public:
	/**
	 * Default constructor.
	 */
	__host__ __device__ MObject(){}

	/**
	 * Default destructor.
	 */
	virtual ~MObject(){}

	__host__ __device__ virtual void callMethod( int functionId, void *arg) = 0;

};

typedef MObject *instantiate_t( void *argument );
typedef void destroy_t( MObject * );

} /* namespace mass */
