/**
 *  @file Dispatcher.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#define COMPUTE_CAPABILITY_MAJOR 3
#include <sstream>
#include <algorithm>  // array compare
#include <iterator>

#include "Dispatcher.h"
#include "cudaUtil.h"
#include "Logger.h"

#include "DeviceConfig.h"
#include "AgentsPartition.h"
#include "Agents.h"
#include "Place.h"
#include "PlacesPartition.h"
#include "Places.h"
#include "DataModel.h"

using namespace std;

namespace mass {

__global__ void callAllPlacesKernel(Place **ptrs, int nptrs, int functionId,
		void *argPtr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < nptrs) {
		ptrs[idx]->callMethod(functionId, argPtr);
	}
}

/**
 * neighbors is converted into a 1D offset of relative indexes before calling this function
 */__global__ void setNeighborPlacesKernel(Place **ptrs, int nptrs,
		int* neighbors, int nNeighbors) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < nptrs) {
		PlaceState *state = ptrs[idx]->getState();
		int offset;
		int skippedNeighbors = 0;
		for (int i = 0; i < nNeighbors; ++i) {
			offset = neighbors[i];
			int j = idx + neighbors[i];
			if (j >= 0 && j < nptrs) {
				state->neighbors[i - skippedNeighbors] = ptrs[j];
				state->inMessages[i- skippedNeighbors] = ptrs[j]->getMessage();
			} else {
				skippedNeighbors++;
			}
		}
	}
}

Dispatcher::Dispatcher() {
	nextDevice = 0;
	model = NULL;
	initialized = false;
	neighborhood = NULL;
	nNeighbors = 0;
}

void Dispatcher::init(int &ngpu) {
	// adapted from the Cuda Toolkit Documentation: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
	if (!initialized) {
		initialized = true;
		Logger::debug(("Initializing Dispatcher"));
		int gpuCount;
		cudaGetDeviceCount(&gpuCount);

		if (0 == ngpu || ngpu > gpuCount) { // use all available GPU resources
			ngpu = gpuCount;
		}

		vector<int> devices;
		for (int device = 0; device < ngpu; ++device) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device);

			Logger::debug("Device %d has compute capability %d.%d", device,
					deviceProp.major, deviceProp.minor);

			if (COMPUTE_CAPABILITY_MAJOR == deviceProp.major) {
				// use this GPU
				devices.push_back(device);
			}
		}

		Logger::debug("Found %d device%s with compute capability %d.X",
				devices.size(), devices.size() > 1 ? "s" : "",
				COMPUTE_CAPABILITY_MAJOR);

		for (int i = 0; i < devices.size(); i++) {
			DeviceConfig d(devices[i]);
			deviceInfo.push_back(d);
		}

		model = new DataModel(devices.size());
	}
}

Dispatcher::~Dispatcher() {
	for (int i = 0; i < deviceInfo.size(); ++i) {
		DeviceConfig &d = deviceInfo[i];
		Logger::debug("Freeing deviceConfig %d", d.deviceNum);
		d.freeDevice();
	}
}

Place** Dispatcher::refreshPlaces(int handle) {
	if (initialized) {
		Logger::debug("Entering Dispatcher::refreshPlaces");

		int stateSize = model->getPlacesModel(handle)->getStateSize();

		map<DeviceConfig*, Partition*>::iterator it = deviceToPart.begin();
		while (it != deviceToPart.end()) {
			PlacesPartition* p = it->second->getPlacesPartition(handle);
			// gets the state belonging to this partition
			void *devPtr = it->first->getPlaceState(handle);
			int qty = p->sizeWithGhosts();
			int bytes = stateSize * qty;
			void *tmp = malloc(bytes);

			// copy the state to the host
			CATCH(cudaMemcpy(tmp, devPtr, bytes, D2H));

			// copy just section results to model
			char *src = (char*) tmp;
			src += stateSize * p->getGhostWidth();
			memcpy(p->getLeftBuffer()->getState(), src, p->size() * stateSize);

			free(tmp);
			++it;
		}

		Logger::debug("Exiting Dispatcher::refreshPlaces");
	}

	return model->getPlacesModel(handle)->getPlaceElements();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument,
		int argSize) {
	if (initialized) {
		Logger::debug("Calling all on places[%d]", placeHandle);

		if (1 == model->getNumPartitions()) {

			int rank = 0;
			Partition* partition = model->getPartition(rank);

			DeviceConfig *d;
			if (0 == partToDevice.count(partition)) { // this partition needs to be loaded
				d = getNextDevice();
				unloadDevice(d);
				d->loadPartition(partition, placeHandle);

				partToDevice[partition] = d;
				deviceToPart[d] = partition;

				Logger::debug("Loaded partition[%d]", placeHandle);
			} else {
				d = partToDevice[partition];
			}

			// load any necessary arguments
			void *argPtr = NULL;
			if (NULL != argument) {
				d->load(argPtr, argument, argSize);
			}

			Logger::debug("Calling callAllPlacesKernel");
			d->setAsActiveDevice();
			PlacesPartition *pPart = partition->getPlacesPartition(placeHandle);
			callAllPlacesKernel<<<pPart->blockDim(), pPart->threadDim()>>>(
					d->getDevPlaces(0), pPart->sizeWithGhosts(), functionId,
					argPtr);
			CHECK();

			if (NULL != argPtr) {
				Logger::debug("Freeing device args.");
				cudaFree(argPtr);
			}

		}
//		else {
//			// TODO in phase 2
//			 execute call on currently loaded partitions
//			 for each rank
//			   for ( int rank = 0; rank < numRanks; ++rank ) {
//			       DeviceConfig d = deviceInfo.front ( );
//			       deviceInfo.pop ( );
//
//			       PlacesPartition *pPart = places->getPartition ( rank );
//			       if ( !pPart->isLoaded ( ) ) {
//			           loadPlacesPartition ( places->getPartition ( rank ), d );
//			       }
//
//			       // load all corresponding agents partitions of the same rank
//			       for ( int handle = 0; handle < Mass::numAgentsInstances ( ); ++handle ) {
//			           Agents *agents = Mass::getAgents ( handle );
//
//			           // there may be more than a single places collection in this simulation
//			           if ( agents->getPlacesHandle ( ) == placeHandle ) {
//			               AgentsPartition* aPart = agents->getPartition ( rank );
//			               if ( !aPart->isLoaded ( ) ) {
//			                   loadAgentsPartition ( aPart, d );
//			               }
//			           }
//			       }
//
//			       // execute the call on the partition
//			       Place** devPlaces = d.devPlaces[ placeHandle ];
//			       callAllPlacesKernel <<<pPart->blockDim ( ), pPart->threadDim ( ), d.inputStream >>>( devPlaces, functionId, devArg, argSize);
//			       __cudaCheckError ( __FILE__, __LINE__ );
//			   }
//		}

		Logger::debug("Exiting Dispatcher::callAllPlaces()");
	}
}

void *Dispatcher::callAllPlaces(int handle, int functionId, void *arguments[],
		int argSize, int retSize) {
	// perform call all
	callAllPlaces(handle, functionId, arguments, argSize);
	// get data from GPUs
	refreshPlaces(handle);
	// get necessary pointers and counts
	int qty = model->getPlacesModel(handle)->getNumElements();
	Place** places = model->getPlacesModel(handle)->getPlaceElements();
	void *retVal = malloc(qty * retSize);
	char *dest = (char*) retVal;

	// TODO handle using OpenMP
	for (int i = 0; i < qty; ++i) {
		// copy messages to a return array
		memcpy(dest, places[i]->getMessage(), retSize);
		dest += retSize;
	}
	return retVal;
}

bool compArr(int* a, int aLen, int *b, int bLen) {
	if (aLen != bLen) {
		return false;
	}

	for (int i = 0; i < aLen; ++i) {
		if (a[i] != b[i])
			return false;
	}
	return true;
}

bool Dispatcher::updateNeighborhood(int handle, vector<int*> *vec) {
	int *offsets = new int[vec->size()];
	PlacesModel *p = model->getPlacesModel(handle);
	int nDims = p->getNumDims();
	int *dimensions = p->getDims();
	int numElements = p->getNumElements();

	// calculate an offset for each neighbor in vec
	for (int j = 0; j < vec->size(); ++j) {
		int *indices = (*vec)[j];
		int offset = 0; // accumulater for row major offset
		int multiplier = 1;

		// a single X will pass over y*z elements,
		// a single Y will pass over z elements, and a Z will pass over 1 element.
		// each dimension will be removed from multiplier before calculating the
		// size of each index's "step"
		for (int i = 0; i < nDims; i++) {
			// convert from raster to cartesian coordinates
			if (1 == i) {
				offset -= multiplier * indices[i];
			} else {
				offset += multiplier * indices[i];
			}

			multiplier *= dimensions[i]; // remove dimension from multiplier
		}
		offsets[j] = offset;
	}

	bool same = compArr(neighborhood, nNeighbors, offsets, vec->size());

	if (!same) {
		delete[] neighborhood;
		neighborhood = offsets;
		nNeighbors = vec->size();
	} else {
		delete[] offsets;
	}

	return same;
}

void Dispatcher::exchangeAllPlaces(int handle, int functionId,
		std::vector<int*> *destinations) {

	updateNeighborhood(handle, destinations);

	// exchange places if necessary
	if ( model->getNumPartitions() == 1) {
		Place** ptrs = deviceInfo[0].getDevPlaces(handle);
		int nptrs = deviceInfo[0].countDevPlaces(handle);
		PlacesPartition *p = model->getPartition(0)->getPlacesPartition(handle);

		// TODO move this to a global params object on GPU
		int *d_nbrs = NULL;
		size_t bytes = sizeof(int) * nNeighbors;
		CATCH(cudaMalloc((void** ) &d_nbrs, bytes));
		CATCH(cudaMemcpy(d_nbrs, neighborhood, bytes,H2D));

		setNeighborPlacesKernel<<<p->blockDim(), p->threadDim()>>>(ptrs, nptrs,
				d_nbrs, nNeighbors);
		CHECK();

		CATCH(cudaFree(d_nbrs));
	}
//		else {
//			// TODO phase II
//		}
//	}
}

void Dispatcher::exchangeBoundaryPlaces(int handle) {
	//TODO issue call in Phase II
}

Agent** Dispatcher::refreshAgents(int handle) {
	// TODO get the unload the slices for this handle from the GPU without deleting
	return NULL;
}

void Dispatcher::callAllAgents(int handle, int functionId, void *argument,
		int argSize) {
	//TODO issue call
}

void *Dispatcher::callAllAgents(int handle, int functionId, void *arguments[],
		int argSize, int retSize) {
	//TODO issue call
	return NULL;
}

void Dispatcher::manageAllAgents(int handle) {
	//TODO issue call
}

DeviceConfig *Dispatcher::getNextDevice() {
	DeviceConfig *d = &deviceInfo[nextDevice];
	nextDevice = (nextDevice + 1) % deviceInfo.size();
	return d;
}

void Dispatcher::unloadDevice(DeviceConfig *device) {
	if (deviceToPart.count(device) > 0) {
		Partition* p = deviceToPart[device];

		map<int, PlacesPartition*> places = p->getPlacesPartitions();
		map<int, PlacesPartition*>::iterator itP = places.begin();
		while (itP != places.end()) {
			refreshPlaces(itP->first);

			// get agents for this place
			map<int, AgentsPartition*> agents = p->getAgentsPartitions(
					itP->first);
			map<int, AgentsPartition*>::iterator itA = agents.begin();
			while (itA != agents.end()) {
				refreshAgents(itA->first);
			}
		}

		partToDevice.erase(p);
		deviceToPart.erase(device);
	}
}

//int ngpu;                   // number of GPUs in use
//std::map<PlacesPartition *, DeviceConfig> loadedPlaces; // tracks which partition is loaded on which GPU
//std::map<AgentsPartition*, DeviceConfig> loadedAgents; // tracks whicn partition is loaded on which GPU
//std::map<int, DeviceConfig> deviceInfo;

}// namespace mass

