
#ifndef PLACESMODEL_H_
#define PLACESMODEL_H_

#define THREADS_PER_BLOCK 512

#include <map>

#include "MassException.h"
#include "Place.h"
#include "PlaceState.h"
#include "Logger.h"

namespace mass {

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

	/**
	 * Returns the ideal block dimension for this PlacesModel. Used for launching
	 * kernel functions on this PlacesModel's data.
	 *
	 * @return
	 */
	dim3 blockDim();

	/**
	 * Returns the ideal thread dimension for this PlacesModel. Used for launching
	 * kernel functions on this PlacesModel's data.
	 *
	 * @return
	 */
	dim3 threadDim();

	template<typename P, typename S>
	static PlacesModel* createPlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

private:

	PlacesModel(int handle, int dimensions, int size[], int qty);

	/**
	 * Refreshes the ideal dimensions for kernel launches. This should be called
	 * only when the PlacesModel is created.
	 */
	void setIdealDims();

	// initialized in creatPlaces function
	Place** places;
	void* state;
	int stateBytes;

	int handle;
	int numDims; // the number of dimensions for this Places_Base (i.e. 1D, 2D, 3D, etc...)
	int *dimensions; // dimensions of the grid in which these places are located. It must be numDims long
	unsigned numElements;

	/*
	 * Dimentions of blocks and threads for GPU
	 * 0 is blockdim, 1 is threaddim
	 */
	dim3 dims[2];
};

template<typename P, typename S>
PlacesModel* PlacesModel::createPlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty) {

	// TODO potential gap if state is not instantiated on the GPU as well. Would not use the argument.
	PlacesModel *p = new PlacesModel(handle, dimensions, size, qty);
	S* tmpPtr = new S[qty];
	p->state = tmpPtr;
	p->stateBytes = sizeof(S);

	p->places = new Place*[qty];
	for (int i = 0; i < qty; ++i) {
		Place *pl = new P((PlaceState*) &(tmpPtr[i]), argument);
		p->places[i] = pl;
	}
	return p;
}

} // end namespace

#endif /* PLACESMODEL_H_ */
