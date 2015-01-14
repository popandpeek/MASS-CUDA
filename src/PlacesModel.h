/*
 *  @file PlacesModel.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef PLACESMODEL_H_
#define PLACESMODEL_H_

#include <map>

#include "PlacesPartition.h"
#include "MassException.h"
#include "PlaceState.h"
#include "PlacesModel.h"

namespace mass {

// forward declarations
class DeviceConfig;
class AgentsModel;

class PlacesModel {

public:

	virtual ~PlacesModel();

	Place** getPlaceElements();
	void* getStatePtr();
	int getStateSize();

	int getHandle();
	int getNumDims();
	int* getDims();
	unsigned getNumElements();
	int getGhostWidth();

	template<typename P, typename S>
	static PlacesModel* createPlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty, int boundary_width);

private:

	PlacesModel(int handle, int dimensions, int size[], int qty, int boundary_width);

	// initialized in creatPlaces function
	Place** places;
	void* state;
	int stateBytes;


	int handle;
	int numDims; // the number of dimensions for this Places_Base (i.e. 1D, 2D, 3D, etc...)
	int *dimensions; // dimensions of the grid in which these places are located. It must be numDims long
	unsigned numElements;
	int boundary_width;
};



template<typename P, typename S>
PlacesModel* PlacesModel::createPlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty, int boundary_width){

	// TODO potential gap if state is not instantiated on the GPU as well. Would not use the argument.
	PlacesModel *p = new PlacesModel(handle, dimensions, size, qty, boundary_width);
	S* tmpPtr = new S[qty];
	p->state = tmpPtr;
	p->stateBytes = sizeof(S);

	p->places = new Place*[qty];
	for(int i = 0; i < qty; ++i){
		Place *pl =new P( (PlaceState*) &(tmpPtr[i]), argument );
		p->places[i] = pl;
	}
	return p;
}

} // end namespace

#endif /* PLACESMODEL_H_ */
