
#ifndef DATAMODEL_H_
#define DATAMODEL_H_

#include <vector>
#include <map>

#include "PlacesModel.h"
#include "AgentsModel.h"
#include "Logger.h"

namespace mass {

class DataModel {

public:
	DataModel();
	~DataModel();

	PlacesModel* getPlacesModel(int handle);
	std::map<int, PlacesModel*> getAllPlacesModels();

	AgentsModel* getAgentsModel(int handle);

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	AgentsModel* instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents);

private:
	void addPlacesModel(PlacesModel *places);
	void addAgentsModel(AgentsModel *agents);
	
	/**
	 * Maps a places handle to a collection.
	 */
	std::map<int, PlacesModel*> placesMap;
	/**
	 * Maps a agents handle to a collection.
	 */
	std::map<int, AgentsModel*> agentsMap;

};

template<typename P, typename S>
PlacesModel* DataModel::instantiatePlaces(int handle, void *argument,
		int argSize, int dimensions, int size[], int qty) {
	Logger::debug("Entering DataModel::instantiatePlaces\n");
	if (placesMap.count(handle) > 0) {
		Logger::warn("A places model with the handle %d already exists", handle);
		return placesMap[handle];
	}

	PlacesModel *p = PlacesModel::createPlaces<P, S>(handle, argument, argSize,
			dimensions, size, qty);
	addPlacesModel(p);
	return p;
}

template<typename AgentType, typename AgentStateType>
AgentsModel* DataModel::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents) {
	Logger::debug("Entering DataModel::instantiateAgents\n");

	if (agentsMap.count(handle) > 0) {
		Logger::warn("A agents model with the handle %d already exists", handle);
		return agentsMap[handle];
	}

	AgentsModel *a = AgentsModel::createAgents<AgentType, AgentStateType> (handle, argument, argSize, nAgents);
	addAgentsModel(a);
	return a;
}

} // end namespace mass

#endif /* DATAMODEL_H_ */
