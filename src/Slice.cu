/**
 *  @file Slice.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#include "cudaUtil.h"
#include "MassException.h"
#include "Model.h"
#include "Slice.h"

namespace mass {

Slice::Slice(int rank) {
	this->rank = rank;
	this->isloaded = false;
}
Slice::~Slice() {
	agents.empty();
	places.empty();
}

bool Slice::addAgentsSlice(AgentsSlice slice) {
	agents[slice.handle] = slice;
	return true;
}

AgentsSlice Slice::getAgents(int handle) {
	std::map<int, AgentsSlice>::iterator it = agents.find(handle);
	if (it == agents.end()) {
		throw MassException("Agents handle not found.");
	}
	return it->second;
}

int Slice::getNumAgents() {
	return agents.size();
}

bool Slice::addPlacesSlice(PlacesSlice slice) {
	places[slice.handle] = slice;
	return true;
}

PlacesSlice Slice::getPlaces(int handle) {
	std::map<int, PlacesSlice>::iterator it = places.find(handle);
	if (it == places.end()) {
		throw MassException("Places handle not found.");
	}
	return it->second;
}

int Slice::getNumPlaces() {
	return places.size();
}

/** load and unload functions. */
void Slice::load(cudaStream_t stream) {
	std::map<int, AgentsSlice>::iterator agent_it = agents.begin();
	while (agent_it != agents.end()) {
		AgentsSlice slice = agent_it->second;
		size_t count = slice.qty * sizeof(slice.begin[0]);
		CATCH(cudaMalloc((void** ) &slice.d_begin, count));
		CATCH(
				cudaMemcpyAsync(slice.d_begin, slice.begin, count,
						cudaMemcpyHostToDevice, stream));
	}

	std::map<int, PlacesSlice>::iterator place_it = places.begin();
	while (place_it != places.end()) {
		PlacesSlice slice = place_it->second;
		size_t count = slice.qty * sizeof(slice.begin[0]);
		CATCH(cudaMalloc((void** ) &slice.d_begin, count));
		CATCH(
				cudaMemcpyAsync(slice.d_begin, slice.begin, count,
						cudaMemcpyHostToDevice, stream));
	}
	isloaded = true;
}

void Slice::retreive(cudaStream_t stream, bool freeOnRetreive) {
	std::map<int, AgentsSlice>::iterator agent_it = agents.begin();
	while (agent_it != agents.end()) {
		AgentsSlice slice = agent_it->second;
		size_t count = slice.qty * sizeof(slice.begin[0]);
		CATCH(
				cudaMemcpyAsync(slice.begin, slice.d_begin, count,
						cudaMemcpyDeviceToHost, stream));
		if (freeOnRetreive) {
			CATCH(cudaFree(slice.d_begin));
		}
	}

	std::map<int, PlacesSlice>::iterator place_it = places.begin();
	while (place_it != places.end()) {
		PlacesSlice slice = place_it->second;
		size_t count = slice.qty * sizeof(slice.begin[0]);
		CATCH(
				cudaMemcpyAsync(slice.begin, slice.d_begin, count,
						cudaMemcpyDeviceToHost, stream));
		if (freeOnRetreive) {
			CATCH(cudaFree(slice.d_begin));
		}
	}
	isloaded = !freeOnRetreive; // if free == true, then isloaded = false
}

bool Slice::isLoaded() {
	return isloaded;
}

int Slice::getRank() {
	return rank;
}

} // end namespace
