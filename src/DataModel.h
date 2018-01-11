
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

	PlacesPartition* getPartition(int handle);
	std::map<int, PlacesPartition*> getAllPlacesPartitions();

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

private:
	void addPlacesModel(PlacesModel *places);

	void partitionPlaces(PlacesModel *places);
	// Partition* partition;
	std::map<int, PlacesPartition*> placesPartitionsByHandle;  // handle to partition map. 

	/**
	 * Maps a places handle to a collection.
	 */
	std::map<int, PlacesModel*> placesMap;

};

template<typename P, typename S>
PlacesModel* DataModel::instantiatePlaces(int handle, void *argument,
		int argSize, int dimensions, int size[], int qty) {
	Logger::debug("Entering DataModel::instantiatePlaces\n");
	if (placesMap.count(handle) > 0) {
		Logger::debug("placesMap.count(handle) > 0\n");  //TODO: replace with warning
		return placesMap[handle];
	}

	PlacesModel *p = PlacesModel::createPlaces<P, S>(handle, argument, argSize,
			dimensions, size, qty);
	addPlacesModel(p);
	return p;
}

} // end namespace mass

#endif /* DATAMODEL_H_ */
