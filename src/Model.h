/**
 *  @file Model.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
//#ifndef MODEL_H_
//#define MODEL_H_
#pragma once
#include <map>
#include <vector>
#include <string>
#include "Slice.h"

namespace mass {
class Agents;
class Places;

struct AgentsSlice {
	Agent *begin; // the first Agent in this slice
	Agent *d_begin; // first agent on device
	int qty; // the number of agents in this slice
	int handle;
};

struct PlacesSlice {
	Place *begin; // the start of this slice of Places, and start of left buffer
	Place *d_begin; // the start of this slice of Places, and start of left buffer
	int qty; // the number of place elements in this slice
	int ghostWidth; // the number of elements to send to another rank when exchanging borders
	int handle;
};

class Model {
private:
	std::map<int, Agents*> agentsMap;
	std::map<int, Places*> placesMap;
	std::map<int, Slice> slicesMap;

public:
	Model();
	~Model();
	bool addAgents(Agents *agents);
	bool addPlaces(Places *places);
	Agents *getAgents(int handle);
	Places *getPlaces(int handle);
	Slice &getSlice(int rank);
	int getNumSlices();
	void setNumSlices(int n);
	void endTurn(); // not yet implemented. Eventually will trigger "clean up" between turns

private:
	void addAgentsToSlices(Agents *agents);

	// dims must be in order { numRows, numCols, zCols, ...}
	// idx must be in order {rowNum, colNum, zNum, ...}
	int toRowMajorIdx(int n, int *dims, int *idx);

	// dims must be in order { numRows, numCols, ...}
	int *toVectorIndex(int n, int *dims, int rmi);

	void addPlacesToSlices(Places *places);

};
// end Model
}// end namespace
//#endif // MODEL_H_
