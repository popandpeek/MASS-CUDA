
#ifndef DATAMODEL_H_
#define DATAMODEL_H_

#include <vector>
#include <map>

#include "PlacesModel.h"
#include "Partition.h"
#include "Logger.h"

namespace mass {

class DataModel {

public:
	DataModel();
	~DataModel();

	PlacesModel* getPlacesModel(int handle);
	std::map<int, PlacesModel*> getAllPlacesModels();

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

private:
	void addPlacesModel(PlacesModel *places);
	
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
