

#include "Partition.h"
#include "Logger.h"
#include "MassException.h"

namespace mass {

Partition::Partition() {}

Partition::~Partition() {
	for (int i = 0; i < placesMap.size(); ++i) {
		delete placesMap[i];
	}
	placesMap.clear();
}

PlacesPartition* Partition::getPlacesPartition(int handle) {

	if (placesMap.count(handle) == 0) {
		Logger::error("There is no handle %d in Partition::placesMap", handle);
		throw MassException("Missing handle");
	}

	return placesMap[handle];
}

std::map<int, PlacesPartition*> Partition::getPlacesPartitions() {
	return placesMap;
}

void Partition::addPlacesPartition(PlacesPartition* places) {
	placesMap[places->getHandle()] = places;
}

} /* namespace mass */
