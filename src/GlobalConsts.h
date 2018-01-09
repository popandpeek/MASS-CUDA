/*
 *  @file GlobalConsts.h
 *  @author Nate Hart
 *	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef GLOBALCONSTS_H_
#define GLOBALCONSTS_H_

#include "Place.h"

namespace mass {

struct GlobalConsts {
	int globalDims[MAX_DIMS];
	int localDims[MAX_DIMS];


	GlobalConsts() {
		memset(globalDims, 0, sizeof(int) * MAX_DIMS);
		memset(localDims, 0, sizeof(int) * MAX_DIMS);
	}
};

} /* namespace mass */
#endif /* GLOBALCONSTS_H_ */
