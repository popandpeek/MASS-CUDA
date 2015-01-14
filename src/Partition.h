/**
 *  @file Partition.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */
#pragma once
#include <map>

#include "PlacesPartition.h"
#include "AgentsPartition.h"

namespace mass {

class Partition {
public:
	Partition(int rank);
	virtual ~Partition();
	int getRank();
//	bool isLoaded();
	PlacesPartition* getPlacesPartition(int handle);
	std::map<int, PlacesPartition*> getPlacesPartitions();
	std::map<int, AgentsPartition*> getAgentsPartitions(int placesHandle);

	void addPlacesPartition(PlacesPartition* places);
	void addAgentsPartition(AgentsPartition* agents);

private:
	int rank;
//	bool loaded;

	// handle to partition map. All partitions have the same rank.
	std::map<int, PlacesPartition*> placesMap;
	std::map<int, AgentsPartition*> agentsMap;
};

} /* namespace mass */
