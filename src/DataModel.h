/*
 *  @file DataModel.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef DATAMODEL_H_
#define DATAMODEL_H_

#include <vector>
#include <map>

#include "PlacesModel.h"
#include "Partition.h"
 #include "Logger.h"

namespace mass {
//class AgentsModel;

class DataModel {

public:
	DataModel();
	~DataModel();

	PlacesModel* getPlacesModel(int handle);

	Partition* getPartition();

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty, int boundary_width);

private:
	void addPlacesModel(PlacesModel *places);

	void partitionPlaces(PlacesModel *places);
	Partition* partition; //replaces the map of partitions

	/**
	 * Maps a places handle to a collection.
	 */
	std::map<int, PlacesModel*> placesMap;

};

template<typename P, typename S>
PlacesModel* DataModel::instantiatePlaces(int handle, void *argument,
		int argSize, int dimensions, int size[], int qty, int boundary_width) {
	Logger::debug("Entering DataModel::instantiatePlaces\n");
	if (placesMap.count(handle) > 0) {
		Logger::debug("placesMap.count(handle) > 0\n");
		return placesMap[handle];
	}

	PlacesModel *p = PlacesModel::createPlaces<P, S>(handle, argument, argSize,
			dimensions, size, qty, boundary_width);
	addPlacesModel(p);
	return p;
}

} // end namespace mass

#endif /* DATAMODEL_H_ */
