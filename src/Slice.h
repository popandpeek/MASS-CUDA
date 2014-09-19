/**
 *  @file Slice.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
//#ifndef SLICE_H_
//#define SLICE_H_
#pragma once
#include <map>
#include <string>

#include "Place.h"

namespace mass {
struct AgentsSlice;
struct PlacesSlice;
/**
 *  This class represents a GPU-sized chunk of the overall data model. This
 *  includes a portion of the Places, as well as all agents residing on those
 *  places.
 */
class Slice {
	int rank;
	bool isloaded;
	std::map<int, AgentsSlice> agents;
	std::map<int, PlacesSlice> places;

public:
	Slice(int rank);
	~Slice();

	bool addAgentsSlice(AgentsSlice slice);
	AgentsSlice getAgents(int handle);
	int getNumAgents();

	bool addPlacesSlice(PlacesSlice slice);
	PlacesSlice getPlaces(int handle);
	int getNumPlaces();

	/** load and unload functions. */
	void load(cudaStream_t stream);
	void retreive(cudaStream_t stream, bool freeOnRetreive = true);
	bool isLoaded();

	int getRank();
};
// end Slice
}// end namespace
//#endif // SLICE_H_
