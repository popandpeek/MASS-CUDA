
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
