#include "PlacesModel.h"
using namespace std;

namespace mass {

PlacesModel::PlacesModel(int handle, int dimensions, int size[], int qty) {
	Logger::debug("Running PlacesModel constructor");
	this->handle = handle;
	this->numElements = qty;
	this->numDims = dimensions;
	this->dimensions = size;
	setIdealDims();
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

dim3 PlacesModel::blockDim() {
	return dims[0];
}

dim3 PlacesModel::threadDim() {
	return dims[1];
}

void PlacesModel::setIdealDims() {
	Logger::debug("Inside PlacesModel::setIdealDims");
	int numBlocks = (numElements - 1) / BLOCK_SIZE + 1;
	dim3 blockDim(numBlocks);

	int nThr = (numElements - 1) / numBlocks + 1;
	dim3 threadDim(nThr);

	dims[0] = blockDim;
	dims[1] = threadDim;
}

} // end namespace
