
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
}

void DataModel::addAgentsModel(AgentsModel *agents) {
	Logger::debug("Entering DataModel::addAgentsModel\n");
	if (NULL == agents) {
		throw MassException("Null pointer in addAgentsModel");
	}

	int handle = agents->getHandle();
	if (agentsMap.count(handle) > 0) {
		Logger::error("DataModel::agentsMap already contains AgentsModel %d",
				handle);
		throw MassException("Adding same collection more than once.");
	}

	agentsMap[handle] = agents;
}

PlacesModel* DataModel::getPlacesModel(int handle) {
	return placesMap[handle];
}

AgentsModel* DataModel::getAgentsModel(int handle) {
	return agentsMap[handle];
}

std::map<int, PlacesModel*> DataModel::getAllPlacesModels() {
	return placesMap;
}

} // end namespace
