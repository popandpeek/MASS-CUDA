/*
 * DataModel.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: nhart
 */

#include "DataModel.h"
#include "Logger.h"

using namespace std;

namespace mass {

DataModel::DataModel(int numPartitions, int boundary_width) {
	nParts = numPartitions;
	for (int i = 0; i < nParts; ++i) {
		Partition* p = new Partition(i);
		partitions[i] = p;
	}
	ghostWidth = boundary_width;
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

//	for (int i = 0; i < agentsMap.size(); ++i) {
//		delete agentsMap[i];
//	}
//	agentsMap.clear();
}

void DataModel::addPlacesModel(PlacesModel *places) {
	if(NULL == places){
		throw MassException("Null pointer in addPlacesModel");
	}

	int handle = places->getHandle();
	if(placesMap.count(handle) > 0){
		Logger::error("DataModel::placesMap already contains PlacesModel %d", handle);
		throw MassException("Adding same collection more than once.");
	}

	placesMap[handle] = places;
	partitionPlaces(places);
}

void DataModel::partitionPlaces(PlacesModel *places) {
	int qty = places->getNumElements();
	int sliceSize = qty / nParts;
	int remainder = qty % nParts;

	Place **elems = places->getPlaceElements();
	for (int rank = 0; rank < nParts; ++rank) {

		int size = sliceSize;
		if(1 != nParts && rank == nParts - 1){
			size = remainder;
		}

		PlacesPartition* p = new PlacesPartition(places->getHandle(), rank,
				size, ghostWidth, places->getNumDims(), places->getDims());

		if (nParts == 1 || rank == 0) {
			// first rank has no left ghost
			p->setSection(elems);
		} else {
			// include left ghost in this partition's section
			p->setSection(elems + sliceSize * rank - p->getGhostWidth());
		}
		// add this partition to the partition of the same rank
		partitions[rank]->addPlacesPartition(p);
	}
}

PlacesModel* DataModel::getPlacesModel(int handle) {
	return placesMap[handle];
}

void DataModel::addAgents(AgentsModel *agents) {
//	agentsMap[agents->getHandle()] = agents;
//	partitionAgents(agents);
}

Partition* DataModel::getPartition(int rank) {
	Partition* retVal = NULL;
	if(partitions.count(rank)>0){
		retVal = partitions[rank];
	}
	return retVal;
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

} // end namespace
