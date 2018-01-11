#include "PlacesModel.h"
//#include <string>

using namespace std;

namespace mass {

PlacesModel::PlacesModel(int handle, int dimensions, int size[], int qty) {
	this->handle = handle;
	this->numElements = qty;
	this->numDims = dimensions;
	this->dimensions = size;
}

PlacesModel::~PlacesModel() {
	for (int i = 0; i < numElements; ++i) {
		delete places[i];
	}
	delete[] places;
	free(state);
}

Place** PlacesModel::getPlaceElements() {
	return places;
}

void* PlacesModel::getStatePtr() {
	return state;
}

int PlacesModel::getStateSize() {
	return stateBytes;
}

int PlacesModel::getHandle() {
	return handle;
}

int PlacesModel::getNumDims() {
	return numDims;
}

int* PlacesModel::getDims() {
	return dimensions;
}

unsigned PlacesModel::getNumElements() {
	return numElements;
}

} // end namespace
