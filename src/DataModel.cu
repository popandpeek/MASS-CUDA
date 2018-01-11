
#include "DataModel.h"

using namespace std;

namespace mass {

DataModel::DataModel() {
	Logger::debug("DataModel constructor running\n");
}

DataModel::~DataModel() {
	for (int i = 0; i < placesMap.size(); ++i) {
		delete placesMap[i];
	}
	placesMap.clear();

	for (int i = 0; i < placesPartitionsByHandle.size(); ++i) {
		delete placesPartitionsByHandle[i];
	}
	placesPartitionsByHandle.clear();
}

void DataModel::addPlacesModel(PlacesModel *places) {
	Logger::debug("Entering DataModel::addPlacesModel\n");
	if (NULL == places) {
		throw MassException("Null pointer in addPlacesModel");
	}

	int handle = places->getHandle();
	if (placesMap.count(handle) > 0) {
		Logger::error("DataModel::placesMap already contains PlacesModel %d",
				handle);
		throw MassException("Adding same collection more than once.");
	}

	placesMap[handle] = places;
	partitionPlaces(places);
}

void DataModel::partitionPlaces(PlacesModel *places) {
	Logger::debug("Entering DataModel::partitionPlaces\n");
	Place **elems = places->getPlaceElements();
	
	PlacesPartition* p = new PlacesPartition(places->getHandle(), 0 /*rank*/,
			places->getNumElements(), places->getNumDims(), places->getDims());

	p->setSection(elems);
	placesPartitionsByHandle[p->getHandle()] = p;
}

PlacesModel* DataModel::getPlacesModel(int handle) {
	return placesMap[handle];
}

PlacesPartition* DataModel::getPartition(int handle) {
	if (placesPartitionsByHandle.count(handle) == 0) {
		Logger::error("There is no handle %d in placesPartitionsByHandle", handle);
		throw MassException("Missing handle");
	}
	return placesPartitionsByHandle[handle];
}

std::map<int, PlacesPartition*> DataModel::getAllPlacesPartitions() {
	return placesPartitionsByHandle;
}

} // end namespace
