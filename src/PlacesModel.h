
#ifndef PLACESMODEL_H_
#define PLACESMODEL_H_

#include <map>
#include <vector>

#include "MassException.h"
#include "cudaUtil.h"
#include "Place.h"
#include "PlaceState.h"
#include "Logger.h"
#include "settings.h"

namespace mass {

class PlacesModel {

public:

	virtual ~PlacesModel();

	std::vector<Place**> getPlaceElements();
	void* getStatePtr(int);
	int getStateSize();
	void setStatePtr(std::vector<void*>);
	int getPlacesStride();
	int getHandle();
	int getNumDims();
	int* getDims();
	unsigned getNumElements();
	int* getGhostPlaceMultiples();

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
			int dimensions, int size[], int qty, int nDevices);

private:

	PlacesModel(int handle, int dimensions, int size[], int qty);

	/**
	 * Refreshes the ideal dimensions for kernel launches. This should be called
	 * only when the PlacesModel is created.
	 */
	void setIdealDims();

	// initialized in createPlaces function
	std::vector<Place**> places;
	std::vector<void*> state;
	int stateBytes;
	unsigned numElements;
	int placesStride;
	int handle;
	int numDims; // the number of dimensions for this Places_Base (i.e. 1D, 2D, 3D, etc...)
	int *dimensions; // dimensions of the grid in which these places are located. It must be numDims long
	int* ghostSpaceMultiple;

	/*
	 * Dimentions of blocks and threads for GPU
	 * 0 is blockdim, 1 is threaddim
	 */
	dim3 dims[2];
};

template<typename P, typename S>
PlacesModel* PlacesModel::createPlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty, int nDevices) {
	Logger::debug("Entering PlacesModel::createPlaces");

	PlacesModel *p = new PlacesModel(handle, dimensions, size, qty);
	p->placesStride = qty / nDevices;
	p->stateBytes = sizeof(S);
	Logger::debug("PlacesModel::createPlaces: nDevices = %d; placesStride = %d, stateBytes = %d", nDevices, p->placesStride, p->stateBytes);
	for (int i = 0; i < nDevices; ++i) {
		Place** p_ptrs = new Place*[p->placesStride];
		S* tmpPtr = new S[p->placesStride];
		for (int j = 0; j < (p->placesStride); ++j) {
			Place *pl = new P((PlaceState*) &(tmpPtr[j]), argument);
			p_ptrs[j] = pl;
		}

		p->places.push_back(p_ptrs);
		p->state.push_back(tmpPtr);
	}

	Logger::debug("Finished PlacesModel::createPlaces");
	return p;
}

} // end namespace

#endif /* PLACESMODEL_H_ */
