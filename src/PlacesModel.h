
#ifndef PLACESMODEL_H_
#define PLACESMODEL_H_

#include <map>

#include "MassException.h"
#include "cudaUtil.h"
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
			int dimensions, int size[], int qty, std::vector<DeviceConfig> devices);

private:

	PlacesModel(int handle, int dimensions, int size[], int qty);

	/**
	 * Refreshes the ideal dimensions for kernel launches. This should be called
	 * only when the PlacesModel is created.
	 */
	void setIdealDims();

	// initialized in createPlaces function
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
Place** PlacesModel::createPlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty, std::vector<DeviceConfig> devices) {
	Logger::debug("Entering PlacesModel::createPlaces");

	// TODO: Should this be calling in a partition? As is, all devices will get same pointer from PlacesModel.
	PlacesModel *p = new PlacesModel(handle, dimensions, size, qty);
	std::vector<DeviceConfig>::iterator itr;
	for (itr = devices.begin(); itr < devices.end(); itr++) {
		cudaSetDevice(itr->getDeviceNum());
		// TODO: handle quantity differently. Get from partition? 
		// Also, store returned ptr from instantiatePlaces to partition and/or PlacesModel
		itr->instantiatePlaces<P, S>(handle, argument, argSize, dimensions, size, qty / devices.size());
	}

	// S* tmpPtr = new S[qty];
	// p->state = tmpPtr;
	// p->stateBytes = sizeof(S);

	// p->places = new Place*[qty];
	// for (int i = 0; i < qty; ++i) {
	// 	Place *pl = new P((PlaceState*) &(tmpPtr[i]), argument);
	// 	p->places[i] = pl;
	// }
	// Logger::debug("Finished PlacesModel::createPlaces");
	return p;
}

} // end namespace

#endif /* PLACESMODEL_H_ */
