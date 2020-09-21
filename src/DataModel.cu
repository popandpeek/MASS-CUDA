
#include "DataModel.h"

using namespace std;

namespace mass {

DataModel::DataModel(int boundaryWidth = 0) {
	Logger::debug("DataModel constructor running\n");
	ghostWidth = boundaryWidth;
}

DataModel::~DataModel() {
	for (int i = 0; i < partitions.size(); ++i) {
		delete partitions[i];
	}

	partitions.clear();

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

Partition* DataModel::getPartition(int rank) {
	Partition* retVal = NULL;
	if (partitions.count(rank) > 0) {
		retVal = partitions[rank];
	}
	return retVal;
}

void DataModel::partitionPlaces(PlacesModel *places) { 
	// Splits up data elements, handles uneven split by placing add'l items in last partition
	// TODO: Need to take a ptr to data and give a ptr to data chunk in PlacesPartition
	int numElems = places->numElements;
	int elems[nParts];
	int sum = 0;
	int avg = numElems / nParts;
	for (int i = 0; i < nParts - 1; i++) {
		elems[i] = avg;
		sum += avg;
	}

	elems[nParts - 1] = numElems - sum;

	for (int i = 0; i < nParts; i++) {
		Partition *p = getPartition(i);
		if (NULL == p->getPlacesPartition(places->handle)) {
			PlacesPartition *pp = new PlacesPartition(places->places + (sizeof(Place*) * avg * i), 
				places->handle, i, elems[i], 0, places->numDims, places->dimensions);
			p->addPlacesPartition(pp);
		}
	}
}

void DataModel::partitionAgents(AgentsModel *agents) {
	// TODO implement
}

void DataModel::setNumPartitions(int numPartitions) {
	if (numPartitions != getNumPartitions() && numPartitions >= 1) {
		nParts = numPartitions;
		// remove old partitions
		for (int i = 0; i < partitions.size(); ++i) {
			delete partitions[i];
		}
		partitions.clear();

		// make new partitions
		for (int i = 0; i < nParts; ++i) {
			Partition* p = new Partition(i);
			partitions[i] = p;
		}

		// add places and agents to those partitions
		map<int, PlacesModel*>::iterator itP = placesMap.begin();
		while (itP != placesMap.end()) {
			partitionPlaces(itP->second);
			++itP;
		}

		// TODO make sure this works with agents structure
//		map<int, AgentsModel*>::iterator itA = agentsMap.begin();
//		while (itA != agentsMap.end()) {
//			partitionAgents(itA->second);
//			++itA;
//		}
	}
}

int DataModel::getNumPartitions() {
	return nParts;
}

/**
 * Gets the agents associated with the specified places handle. The map key is
 * the Agents handle, and the value is the vector of partitions for that Agents
 * collection.
 *
 * @param placesHandle the handle of the Places collection whose Agents you want
 * to retreive.
 *
 * @return a map of vectors where each vector represents a complete Agents collection.
 */
map<int, AgentsModel*> DataModel::getAgentsForPlace(int placesHandle) {
	map<int, AgentsModel*> retVal;

//	map<int, AgentsPartition*>::iterator itA = agentsMap.begin();
//	while (itA != agentsMap.end()) {
//		if(itA->second->getPlacesHandle() == placesHandle){
//			retVal[itA->second->getHandle()] = itA->second;
//		}
//		++itA;
//	}
	return retVal;
}

PlacesModel* DataModel::getPlacesModel(int handle) {
	return placesMap[handle];
}

AgentsModel* DataModel::getAgentsModel(int handle) {
	return agentsMap[handle];
}

// std::map<int, PlacesModel*> DataModel::getAllPlacesModels() {
// 	return placesMap;
// }

} // end namespace
