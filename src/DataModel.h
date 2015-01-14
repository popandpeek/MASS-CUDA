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
//#include "AgentsModel.h"
#include "Partition.h"

namespace mass {

class AgentsModel;
// TODO remove when class is created

class DataModel {

public:
	DataModel(int numPartitions = 1, int boundary_width = 0);
	~DataModel();

	PlacesModel* getPlacesModel(int handle);

	void setNumPartitions(int numPartitions);
	int getNumPartitions();

	/**
	 * Gets the agents associated with the specified places handle. The map key is
	 * the Agents handle, and the value is the map of Agents.
	 *
	 * @param placesHandle the handle of the Places collection whose Agents you want
	 * to retreive.
	 *
	 * @return a map of vectors where each vector represents a complete Agents collection.
	 */
	std::map<int, AgentsModel*> getAgentsForPlace(int placesHandle);

	Partition* getPartition(int rank);

	template<typename P, typename S>
	PlacesModel* instantiatePlaces(int handle, void *argument, int argSize,
			int dimensions, int size[], int qty, int boundary_width);

private:

	void addPlacesModel(PlacesModel *places);

	void partitionPlaces(PlacesModel *places);

	void addAgents(AgentsModel *agents);

	void partitionAgents(AgentsModel *agents);

	int nParts; /**< The number of partitions in this data model.*/
	int ghostWidth; /**< The width of the boundary.*/

	std::map<int, Partition*> partitions;

	/**
	 * Maps a places handle to a collection.
	 */
	std::map<int, PlacesModel*> placesMap;

	std::map<int, AgentsModel*> agentsMap;

};

template<typename P, typename S>
PlacesModel* DataModel::instantiatePlaces(int handle, void *argument, int argSize,
		int dimensions, int size[], int qty, int boundary_width) {
	PlacesModel *p = PlacesModel::createPlaces<P, S>(handle, argument, argSize,
			dimensions, size, qty, boundary_width);
	addPlacesModel(p);
	return p;
}

} // end namespace

#endif /* DATAMODEL_H_ */
