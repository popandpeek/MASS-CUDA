/**
 *  @file Dispatcher.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#define COMPUTE_CAPABILITY_MAJOR 3
#include <sstream>
#include "Dispatcher.h"
#include "cudaUtil.h"
#include "Logger.h"

#include "DeviceConfig.h"
#include "AgentsPartition.h"
#include "Agents.h"
#include "Place.h"
#include "PlacesPartition.h"
#include "Places.h"

using namespace std;

namespace mass {

__global__ void setPlacePtrsKernel(Place **ptrs, void *objs, int nPtrs,
		int nObjs, int Tsize) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < nPtrs) {
		if (idx < nObjs) {
			char* dest = ((char*) objs) + idx * Tsize;
			ptrs[idx] = (Place*) dest;
		} else {
			ptrs[idx] = NULL;
		}
	}
}

__global__ void callAllPlacesKernel(Place **ptrs, int nptrs, int functionId,
		void *argPtr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < nptrs) {
		ptrs[idx]->callMethod(functionId, argPtr);
	}
}

Dispatcher::Dispatcher() {
	nextDevice = 0;
}

void Dispatcher::init(int &ngpu) {
	// adapted from the Cuda Toolkit Documentation: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

	Logger::debug(("Initializing Dispatcher"));
	int allgpus;
	cudaGetDeviceCount(&allgpus);

	if (0 == ngpu) { // use all available GPU resources
		ngpu = allgpus;
	}

	vector<int> devices;
	for (int device = 0; device < ngpu && device < allgpus; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);

		Logger::debug("Device %d has compute capability %d.%d", device,
				deviceProp.major, deviceProp.minor);

		if (COMPUTE_CAPABILITY_MAJOR == deviceProp.major) {
			// use this GPU
			devices.push_back(device);
		}
	}

	Logger::debug("Found %d device(s) with compute capability %d.X",
			devices.size(), COMPUTE_CAPABILITY_MAJOR);

	for (int i = 0; i < devices.size(); i++) {
		DeviceConfig d(devices[i]);
		deviceInfo.push_back(d);
	}
}

Dispatcher::~Dispatcher() {
	for (int i = 0; i < deviceInfo.size(); ++i) {
		DeviceConfig &d = deviceInfo[i];
		Logger::debug("Freeing deviceConfig %d", d.deviceNum);
		d.freeDevice();
	}
}

void Dispatcher::refreshPlaces(Places *places) {
	// TODO get the unload the slices for this handle from the GPU without deleting
	Logger::debug("Entering Dispatcher::refreshPlaces");
	for (int i = 0; i < places->getNumPartitions(); ++i) {
		PlacesPartition *part = places->getPartition(i);
		if (part->isLoaded()) {
			Logger::debug("PlacesPartition[%d] is loaded", i);
			getPlacesPartition(part, false);
		}
	}

	Logger::debug("Exiting Dispatcher::refreshPlaces");
}

void Dispatcher::callAllPlaces(Places *places, int functionId, void *argument,
		int argSize) {
	Logger::debug("Entering Dispatcher::callAllPlaces()");
	int placeHandle = places->getHandle();
	Logger::debug("Calling all on places[%d]", placeHandle);
	// TODO execute call on currently loaded partitions

	int numRanks = places->getNumPartitions();

	Logger::debug("There are %d ranks", numRanks);
	if (1 == numRanks) {
		DeviceConfig *d = getNextDevice();

		int rank = 0;
		PlacesPartition *pPart = places->getPartition(rank);
		if (!pPart->isLoaded()) {

			Logger::debug("Loaded partition[%d]", placeHandle);
			loadPlacesPartition(pPart, d);

//			// load all corresponding agents partitions of the same rank
//			for (int handle = 0; handle < Mass::numAgentsInstances();
//					++handle) {
//
//				Agents *agents = Mass::getAgents(handle);
//
//				// there may be more than a single places collection in this simulation
//				if (agents->getPlacesHandle() == placeHandle) {
//					Logger::debug("Loading agents rank %d", handle);
//					AgentsPartition* aPart = agents->getPartition(rank);
//					if (!aPart->isLoaded()) {
//						loadAgentsPartition(aPart, d);
//					}
//				}
//			}
		} else {
			d = loadedPlaces[pPart];
		}

		d->setAsActiveDevice();
		// execute the call on the partition
		void *argPtr = NULL;
		if (NULL != argument) {

			Logger::debug("Loading the argument of argSize %d", argSize);
			cudaMalloc((void**) argPtr, argSize);
			cudaMemcpy(argPtr, argument, argSize, H2D);
		}

		Logger::debug("Calling callAllPlacesKernel");
		callAllPlacesKernel<<<pPart->blockDim(), pPart->threadDim()>>>(d->getPlaces(0), pPart->sizePlusGhosts(),
				functionId, argPtr);
		CHECK();

		if (NULL != argPtr) {
			Logger::debug("Freeing device args.");
			cudaFree(argPtr);
		}

	} else {
		// TODO in phase 2
	}
	// for each rank
	//   for ( int rank = 0; rank < numRanks; ++rank ) {
	//       DeviceConfig d = deviceInfo.front ( );
	//       deviceInfo.pop ( );

	//       PlacesPartition *pPart = places->getPartition ( rank );
	//       if ( !pPart->isLoaded ( ) ) {
	//           loadPlacesPartition ( places->getPartition ( rank ), d );
	//       }

	//       // load all corresponding agents partitions of the same rank
	//       for ( int handle = 0; handle < Mass::numAgentsInstances ( ); ++handle ) {
	//           Agents *agents = Mass::getAgents ( handle );

	//           // there may be more than a single places collection in this simulation
	//           if ( agents->getPlacesHandle ( ) == placeHandle ) {
	//               AgentsPartition* aPart = agents->getPartition ( rank );
	//               if ( !aPart->isLoaded ( ) ) {
	//                   loadAgentsPartition ( aPart, d );
	//               }
	//           }
	//       }

	//       // execute the call on the partition
	//       Place** devPlaces = d.devPlaces[ placeHandle ];
	//       callAllPlacesKernel <<<pPart->blockDim ( ), pPart->threadDim ( ), d.inputStream >>>( devPlaces, functionId, devArg, argSize);
	//       __cudaCheckError ( __FILE__, __LINE__ );
	//   }
	Logger::debug("Exiting Dispatcher::callAllPlaces()");
}

void *Dispatcher::callAllPlaces(Places *places, int functionId,
		void *arguments[], int argSize, int retSize) {
	// TODO issue call
	return NULL;
}

void Dispatcher::exchangeAllPlaces(Places *places, int functionId,
		std::vector<int*> *destinations) {
	// TODO issue call
}

void Dispatcher::exchangeBoundaryPlaces(Places *places) {
	//TODO issue call
}

void Dispatcher::refreshAgents(Agents *agents) {
	// TODO get the unload the slices for this handle from the GPU without deleting
}

void Dispatcher::callAllAgents(Agents *agents, int functionId, void *argument,
		int argSize) {
	//TODO issue call
}

void *Dispatcher::callAllAgents(Agents *agents, int functionId, void *arguments[],
		int argSize, int retSize) {
	//TODO issue call
	return NULL;
}

void Dispatcher::manageAllAgents(Agents *agents) {
	//TODO issue call
}

void Dispatcher::loadPlacesPartition(PlacesPartition *part, DeviceConfig *d) {
	d->setAsActiveDevice();
	loadedPlaces[part] = d;

	// load partition onto device
	void* dPtr = part->devicePtr();
	int numPlaces = part->sizePlusGhosts();
	d->setNumPlaces(numPlaces);
	int Tsize = part->getPlaceBytes();
	int numBytes = Tsize * numPlaces;

	// there may be rare edge cases where memory is already allocated
	if (!part->isLoaded() || NULL == dPtr) {
		cudaMalloc((void**) &dPtr, numBytes);

		// update model state
		part->setDevicePtr(dPtr);
		part->setLoaded(true);
	}
	CATCH(
			cudaMemcpy(dPtr, part->hostPtrPlusGhosts(), numBytes, H2D));//;, d.inputStream));

	// TODO set pointer array on GPU
	int bDim, tDim;
	bDim = (numPlaces - 1) / BLOCK_SIZE + 1;
	tDim = (numPlaces - 1) / bDim + 1;

	setPlacePtrsKernel<<<bDim, tDim>>>(d->getPlaces(0), dPtr,
			d->getNumPlacePtrs(0), numPlaces, Tsize);
	CHECK();
}

void Dispatcher::getPlacesPartition(PlacesPartition *part,
		bool freeOnRetrieve) {
	Logger::info("Entering Dispatcher::getPlacesPartition()");
	map<PlacesPartition*, DeviceConfig*>::iterator it = loadedPlaces.find(part);
	if (it == loadedPlaces.end()) {
		Logger::error(
				"Unable to find partition %d. getPlacesPartition aborted.",
				part->getRank());
		return;
	}

	DeviceConfig *d = loadedPlaces[part];
	cudaSetDevice(d->deviceNum);

	// get partition onto device
	Logger::debug("Getting data from device %d.", d->deviceNum);
	char* dPtr = (char*) part->devicePtr(); // again, use of char* to allow pointer arithmitic
	dPtr += part->getPlaceBytes() * part->getGhostWidth(); // we don't want to copy out bad ghost data
	int numBytes = part->getPlaceBytes() * part->size();
//	CATCH(cudaMemcpyAsync(part->hostPtr(), dPtr, numBytes, D2H,
//			d.outputStream));

	CATCH(cudaMemcpy(part->hostPtr(), dPtr, numBytes, D2H));

	if (freeOnRetrieve) {
		Logger::info("Freeing partition rank %d from device memory.",
				part->getRank());
		// update model state
		part->setDevicePtr(NULL);
		part->setLoaded(false);

		cudaFree(part->devicePtr());
		loadedPlaces.erase(part);
	}
	Logger::info("Exiting Dispatcher::getPlacesPartition()");
}

void Dispatcher::loadAgentsPartition(AgentsPartition *part, DeviceConfig *d) {
	d->setAsActiveDevice();
	loadedAgents[part] = d;

	// load partition onto device
	void* dPtr = part->devicePtr();
	int numBytes = part->getPlaceBytes() * part->sizePlusGhosts();

	// there may be rare edge cases where memory is already allocated
	if (!part->isLoaded() || NULL == dPtr) {
		cudaMalloc((void**) &dPtr, numBytes);

		// update model state
		part->setDevicePtr(dPtr);
		part->setLoaded(true);
	}
	cudaMemcpy(dPtr, part->hostPtrPlusGhosts(), numBytes, H2D);
			//,d.inputStream);

	// TODO set pointer array on GPU
}

void Dispatcher::getAgentsPartition(AgentsPartition *part,
		bool freeOnRetrieve) {
	DeviceConfig *d = loadedAgents[part];
	d->setAsActiveDevice();

	// get partition onto device
	char* dPtr = (char*) part->devicePtr(); // again, use of char* to allow pointer arithmitic
	dPtr += part->getPlaceBytes() * part->getGhostWidth(); // we don't want to copy out bad ghost data
	int numBytes = part->getPlaceBytes() * part->size();
	cudaMemcpy(part->hostPtr(), dPtr, numBytes, cudaMemcpyDeviceToHost);//,
			//, d.outputStream);

	if (freeOnRetrieve) {
		// update model state
		part->setDevicePtr(NULL);
		part->setLoaded(false);

		cudaFree(part->devicePtr());
		loadedAgents.erase(part);
	}
}

void Dispatcher::configurePlaces(Places *places) {

	// determine simulation size, if small enough, run on one GPU
	places->setPartitions(1);
//	loadPlacesPartition(places->getPartition(0), getNextDevice());


	// if possible, run simulation one partition per GPU
	// TODO phase II
	// else create n GPU-sized partitions
	// TODO phase III
}

void Dispatcher::configureAgents(Agents *agents) {
	// TODO implement
}

DeviceConfig *Dispatcher::getNextDevice() {
	DeviceConfig *d = &deviceInfo[nextDevice];
	nextDevice = (nextDevice + 1) % deviceInfo.size();
	return d;
}

//int ngpu;                   // number of GPUs in use
//std::map<PlacesPartition *, DeviceConfig> loadedPlaces; // tracks which partition is loaded on which GPU
//std::map<AgentsPartition*, DeviceConfig> loadedAgents; // tracks whicn partition is loaded on which GPU
//std::map<int, DeviceConfig> deviceInfo;

}// namespace mass

