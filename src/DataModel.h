
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
	DataModel(int);
	~DataModel();

	PlacesModel* getPlacesModel(int handle);
	std::map<int, PlacesModel*> getAllPlacesModels();

	AgentsModel* getAgentsModel(int handle);

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	AgentsModel* instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int maxAgents, int* nAgentsDev);


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

	// number of devices
	int nDevices;

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
			dimensions, size, qty, this->nDevices);
	addPlacesModel(p);
	return p;
}

template<typename AgentType, typename AgentStateType>
AgentsModel* DataModel::instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents, int maxAgents, int* nAgentsDev) {
	Logger::debug("Entering DataModel::instantiateAgents\n");

	if (agentsMap.count(handle) > 0) {
		Logger::warn("A agents model with the handle %d already exists", handle);
		return agentsMap[handle];
	}

	AgentsModel *a = AgentsModel::createAgents<AgentType, AgentStateType> (handle, argument, argSize, nAgents, 
			nAgentsDev, maxAgents, this->nDevices);
	addAgentsModel(a);
	return a;
}

} // end namespace mass

#endif /* DATAMODEL_H_ */
