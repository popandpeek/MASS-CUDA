/*
 *  @file Partition.cpp
 *  @author Nate Hart
 *	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Partition.h"
#include "Logger.h"
#include "MassException.h"

namespace mass {

Partition::Partition(int rank) :
		rank(rank) {
}

Partition::~Partition() {
	for (int i = 0; i < placesMap.size(); ++i) {
		delete placesMap[i];
	}
	placesMap.clear();

	//	for (int i = 0; i < agentsMap.size(); ++i) {
	//		delete agentsMap[i];
	//	}
	//	agentsMap.clear();
}

int Partition::getRank() {
	return rank;
}

PlacesPartition* Partition::getPlacesPartition(int handle) {

	if (placesMap.count(handle) == 0) {
		Logger::error("There is no handle %d in Partition::placesMap", handle);
		throw MassException("Missing handle");
	}

	return placesMap[handle];
}

std::map<int, PlacesPartition*> Partition::getPlacesPartitions() {
	return placesMap;
}

std::map<int, AgentsPartition*> Partition::getAgentsPartitions(
		int placesHandle) {
	return agentsMap;
}

void Partition::addPlacesPartition(PlacesPartition* places) {
	placesMap[places->getHandle()] = places;
}

void Partition::addAgentsPartition(AgentsPartition* agents) {
	agentsMap[agents->getHandle()] = agents;
}

} /* namespace mass */
