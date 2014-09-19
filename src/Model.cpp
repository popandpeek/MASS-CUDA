/**
 *  @file Model.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "math.h" // ciel
#include "MassException.h"
#include "Model.h"

namespace mass {

void Model::addAgentsToSlices(Agents<Agent> *agents) {
	Agent* elements = agents->agents; // friend access
	int size = (double) agents->nAgents();
	int numSlices = this->getNumSlices();
	int sliceSize = (int) ceil(((double) size) / numSlices);
	int remainder = size - sliceSize * (numSlices - 1);

	int rank = 0; // first n-1 ranks get a full slice
	for (; rank < size - 1; ++rank) {
		Slice &slice = slicesMap.find(rank)->second;
		AgentsSlice a;
		a.begin = elements + rank * sliceSize;
		a.d_begin = NULL;
		a.qty = remainder;
		a.handle = agents->getHandle();
		slice.addAgentsSlice(a);
	}

	if (0 != rank) { // special case needed for single rank
		++rank;
	}

	// last rank gets remainder
	Slice &slice = slicesMap.find(rank)->second;
	AgentsSlice a;
	a.begin = elements + rank * sliceSize;
	a.d_begin = NULL;
	a.qty = remainder;
	a.handle = agents->getHandle();
	slice.addAgentsSlice(a);
}

// dims must be in order { numRows, numCols, zCols, ...}
// idx must be in order {rowNum, colNum, zNum, ...}
int Model::toRowMajorIdx(int n, int *dims, int *idx) {
	int row_major_index = 0;
	int stride = 1; // tracks how far to increase index
	// stride tracks the "jump" between elements of current dimension
	for (int i = 0; i < n; ++i) {
		stride *= dims[i];
	}

	for (int i = 0; i < n; ++i) {
		// we never want to include current dimension in stride
		// in a 1D array, the stride of numRows is always 1
		stride /= dims[i];
		row_major_index += idx[i] * stride;
	}
	return row_major_index;
}

// dims must be in order { numRows, numCols, ...}
int *Model::toVectorIndex(int n, int *dims, int rmi) {
	int *index = new int[n];
	for (int i = n - 1; i >= 0; --i) {
		index[i] = rmi % dims[i];
		rmi /= dims[i];
	}
	return index;
}

void Model::addPlacesToSlices(Places<Place> *places) {
	Place *elements = places->elements; // friend access
	int n = places->numDims;
	int *dimensions = places->dimensions;
	int size = 1;
	for (int i = 0; i < n; ++i) {
		size *= dimensions[i];
	}

	int numSlices = this->getNumSlices();
	if (1 == numSlices) { // special case for a single slice
		Slice &slice = slicesMap.find(0)->second;
		PlacesSlice p;
		p.begin = elements;
		p.d_begin = NULL;
		p.qty = size;
		p.ghostWidth = 0;
		p.handle = places->getHandle();
		slice.addPlacesSlice(p);
	} else {
		int sliceSize = (int) ceil(((double) size) / numSlices);
		int remainder = size - sliceSize * (numSlices - 1);
		int ghostWidth = size / dimensions[0] * places->boundary_width;

		int rank = 0; // first n-1 ranks get a full slice
		for (; rank < size - 1; ++rank) {
			Slice &slice = slicesMap.find(rank)->second;
			PlacesSlice p;
			p.begin = elements + rank * sliceSize;
			p.d_begin = NULL;
			p.qty = sliceSize;
			p.ghostWidth = ghostWidth;
			p.handle = places->getHandle();
			slice.addPlacesSlice(p);
		}

		// last rank gets remainder elements
		++rank;
		Slice &slice = slicesMap.find(rank)->second;
		PlacesSlice p;
		p.begin = elements + rank * sliceSize;
		p.d_begin = NULL;
		p.qty = remainder;
		p.ghostWidth = ghostWidth;
		p.handle = places->getHandle();
		slice.addPlacesSlice(p);
	}
}

Model::Model() {
	// for the time being, there is only ever one slice
	setNumSlices(1);
}

Model::~Model() {
	agentsMap.empty();
	placesMap.empty();
	slicesMap.empty();
}

bool Model::addAgents(Agents<Agent> *agents) {
	bool isNew = true;
	int handle = agents->getHandle();

	std::map<int, Agents<Agent>*>::iterator it = agentsMap.find(handle);
	if (it != agentsMap.end()) {
		isNew = false;
	}

	if (isNew) {
		agentsMap.insert(std::pair<int, Agents<Agent>*>(handle, agents));
		addAgentsToSlices(agents);  // update the data model
	}
	return isNew;
}

bool Model::addPlaces(Places<Place> *places) {
	bool isNew = true;
	int handle = places->getHandle();

	std::map<int, Places<Place>*>::iterator it = placesMap.find(handle);
	if (it != placesMap.end()) {
		isNew = false;
	}

	if (isNew) {
		placesMap.insert(std::pair<int, Places<Place>*>(handle, places));
		addPlacesToSlices(places); // update the data model
	}
	return isNew;
}

Agents<Agent> *Model::getAgents(int handle) {
	Agents<Agent> *agents = NULL;
	// TODO extract current agents data from GPU
	std::map<int, Agents<Agent>*>::iterator it = agentsMap.find(handle);
	if (it != agentsMap.end()) {
		agents = it->second;
	}
	return agents;
}

Places<Place> *Model::getPlaces(int handle) {
	Places<Place> *places = NULL;
	// TODO extract current places data from GPU
	std::map<int, Places<Place>*>::iterator it = placesMap.find(handle);
	if (it != placesMap.end()) {
		places = it->second;
	}
	return places;
}

int Model::getNumSlices() {
	return slicesMap.size();
}

void Model::setNumSlices(int n) { // not yet implemented

	if (n < 1) {
		throw MassException("Number of slices must be at least 1");
	}

	// temporary code: remove once re-slicing logic is in this function
	if (getNumSlices() > 0) {
		return;
	}

	for (int i = 0; i < n; ++i) {
		std::map<int, Slice>::iterator it = slicesMap.find(i);
		if (it == slicesMap.end()) {
			Slice slice(i);
			slicesMap.insert(std::pair<int, Slice>(i, slice));
		}
	}
}

Slice &Model::getSlice(int rank) {
	std::map<int, Slice>::iterator it = slicesMap.find(rank);
	if (it == slicesMap.end()) {
		throw MassException("There is no slice with that rank");
	}
	return it->second;
}

void Model::endTurn() {
} // not yet implemented

} // end namespace
