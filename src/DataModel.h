
#ifndef DATAMODEL_H_
#define DATAMODEL_H_

#include <vector>
#include <map>

#include "DeviceConfig.h"
#include "PlacesModel.h"
#include "AgentsModel.h"
#include "Partition.h"
#include "Logger.h"

namespace mass {

class DataModel {

public:
	DataModel(std::vector<DeviceConfig> devices, int boundary_width = 0);
	~DataModel();

	PlacesModel* getPlacesModel(int handle);
	//std::map<int, PlacesModel*> getAllPlacesModels();
	AgentsModel* getAgentsModel(int handle);
	void setNumPartitions(int numPartitions);
	int getNumPartitions();
	Partition* getPartition(int rank);

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty);

	template<typename AgentType, typename AgentStateType>
	AgentsModel* instantiateAgents (int handle, void *argument, 
		int argSize, int nAgents);

private:
	int nParts; /**< The number of partitions in this data model.*/
	int ghostWidth; /**< The width of the boundary.*/
	std::vector<DeviceConfig> devices;
	// std::vector<int> devices;
	std::map<int, Partition*> partitions;

	void addPlacesModel(PlacesModel *places);
	void addAgentsModel(AgentsModel *agents);
	void partitionPlaces(PlacesModel *places);
	void partitionAgents(AgentsModel *agents);
	
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
			dimensions, size, qty, devices);
	addPlacesModel(p);
	// partitionPlaces(p);
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

	AgentsModel *a = AgentsModel::createAgents<AgentType, AgentStateType> (handle, argument, 
		argSize, nAgents, devices);
	addAgentsModel(a);
	// partitionAgents(a);
	return a;
}

} // end namespace mass

#endif /* DATAMODEL_H_ */
