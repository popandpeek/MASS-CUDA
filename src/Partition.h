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

namespace mass {

class Partition {
public:
	Partition();
	virtual ~Partition();
	PlacesPartition* getPlacesPartition(int handle);
	std::map<int, PlacesPartition*> getPlacesPartitions();

	void addPlacesPartition(PlacesPartition* places);

private:
    
	std::map<int, PlacesPartition*> placesMap;  // handle to partition map. 
};

} /* namespace mass */
