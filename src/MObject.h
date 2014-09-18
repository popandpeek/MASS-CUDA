/**
 *  @file MObject.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#ifndef MOBJECT_H_
#define MOBJECT_H_

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
	__host__ __device__ MObject( );

	/**
	 * Default destructor.
	 */
	virtual ~MObject( );

	/**
	 * Compares two MObjects to see if they are equal. In the basic
	 * implementation, the pointer values are compared to see if the two
	 * objects being compared are the SAME object. 
	 *
	 * @param rhs the other MObject to which this object shall be compared
	 * @return <code>true</code> if objects are equal
	 */
	__host__ __device__ virtual bool operator==( const MObject &rhs );

	/**
	 * Compares two MObjects to see if they are not equal. In the basic
	 * implementation, the pointer values are compared to see if the two
	 * objects being compared are the SAME object. This simply returns
	 * ! (*this) == rhs. Thus overriding operator== will also change the
	 * behavior of this function as well.
	 *
	 * @param rhs the other MObject to which this object shall be compared
	 * @return <code>true</code> if objects are not equal
	 */
	__host__ __device__ bool operator!=( const MObject &rhs );
	
};

} /* namespace mass */
#endif /* MOBJECT_H_ */
