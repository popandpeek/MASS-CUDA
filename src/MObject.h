/**
 *  @file MObject.h
 *  @author Prof. Fukuda, modified by Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once
#include <cuda_runtime.h>
// easier for end users to understand than the __host__ __device__ meaning.
#define MASS_FUNCTION __host__ __device__

namespace mass {

/**
 * This is the base object for all user-defined objects in the MASS library.
 * Much of the base functionality of objects is defined here and inherited for
 * ease of use. It also serves to define the dllClass creation and destruction
 * functions.
 */
class MObject {
public:

	/**
	 * Default constructor.
	 */
	MASS_FUNCTION MObject(){}

	/**
	 * Default destructor.
	 */
	MASS_FUNCTION virtual ~MObject(){}

};

typedef MObject *instantiate_t( void *argument );
typedef void destroy_t( MObject * );

} /* namespace mass */
