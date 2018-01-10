/*
 * DataModel.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: nhart
 */

#include "DataModel.h"

using namespace std;

namespace mass {

DataModel::DataModel() {
	Logger::debug("DataModel constructor running\n");
	partition = new Partition();
}

DataModel::~DataModel() {
	delete partition;
	for (int i = 0; i < placesMap.size(); ++i) {
		delete placesMap[i];
	}
	placesMap.clear();
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
	partition->addPlacesPartition(p);
}

PlacesModel* DataModel::getPlacesModel(int handle) {
	return placesMap[handle];
}

Partition* DataModel::getPartition() {
	return partition;
}

} // end namespace
