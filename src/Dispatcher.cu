/**
 *  @file Dispatcher.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#define COMPUTE_CAPABILITY_MAJOR 3
#include <sstream>
#include "Agents.h"
#include "AgentsPartition.h"
#include "cudaUtil.h"
#include "Dispatcher.h"
#include "Places.h"
#include "PlacesPartition.h"
#include "Mass.h"

using namespace std;

namespace mass {

Dispatcher::Dispatcher() {
	// do nothing
}

void Dispatcher::init(int ngpu) {
	// adapted from the Cuda Toolkit Documentation: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
	stringstream ss;
	Mass::log(("Initializing Dispatcher"));
	if (0 == ngpu) { // use all available GPU resources
		cudaGetDeviceCount(&ngpu);
	}
	vector<int> devices;
	for (int device = 0; device < ngpu; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n", device,
				deviceProp.major, deviceProp.minor);
		if (COMPUTE_CAPABILITY_MAJOR == deviceProp.major) {
			// use this GPU
			devices.push_back(device);
		}
	}

	for (int i = 0; i < devices.size(); i++) {
		DeviceConfig d(devices[i]);
//        d.deviceNum = devices[i];
//        cudaSetDevice ( d.deviceNum );
//        cudaStreamCreate ( &d.inputStream );
//        cudaStreamCreate ( &d.outputStream );
//        cudaEventCreate ( &d.deviceEvent );
		deviceInfo.push(d);
	}
}

Dispatcher::~Dispatcher() {
	deviceInfo.empty();
//    while(deviceInfo.size()>0) {
//        DeviceConfig d = deviceInfo.front();
//        deviceInfo.pop();
//        cudaSetDevice ( d.deviceNum );
//        // destroy streams
//        cudaStreamDestroy ( d.inputStream );
//        cudaStreamDestroy ( d.outputStream );
//        // destroy events
//        cudaEventDestroy ( d.deviceEvent );
//    }
}

void Dispatcher::refreshPlaces(Places *places) {
	// TODO get the unload the slices for this handle from the GPU without deleting
	for (int i = 0; i < places->getNumPartitions(); ++i) {
		PlacesPartition *part = places->getPartition(i);
		if (part->isLoaded()) {
			DeviceConfig d = loadedPlaces[part];
			cudaSetDevice(d.deviceNum);
//			part->retrieve(d.outputStream, false); // retreive via secondary stream without deleting
		}
	}
}

//__global__ void setPlacePointers ( Places** devPlaces, void* placeOjects, int numPointers, int numPlaces, int Tsize ) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if ( idx < numPointers ) {
//        if ( idx < numPlaces ) {
//            char* ptr = ( char* ) placeObjects;
//            ptr += Tsize * idx;
//            devPlaces[ idx ] = ( Places* ) ptr;
//        } else {
//            devPlaces[ idx ] = NULL;
//        }
//    }
//}
//
//__global__ void callAllPlacesKernel ( Places** devPlaces, int funcID, void *devArg, int argSize ) {
//
//}

void Dispatcher::callAllPlaces(Places *places, int functionId, void *argument,
		int argSize) {
	int placeHandle = places->getHandle();
	// TODO execute call on currently loaded partitions

	int numRanks = places->getNumPartitions();
	if (1 == numRanks) {
		DeviceConfig d = deviceInfo.front();
		deviceInfo.pop();

		int rank = 0;
		PlacesPartition *pPart = places->getPartition(rank);
		if (!pPart->isLoaded()) {
			loadPlacesPartition(pPart, d);
		}

		// load all corresponding agents partitions of the same rank
		for (int handle = 0; handle < Mass::numAgentsInstances(); ++handle) {
			Agents *agents = Mass::getAgents(handle);

			// there may be more than a single places collection in this simulation
			if (agents->getPlacesHandle() == placeHandle) {
				AgentsPartition* aPart = agents->getPartition(rank);
				if (!aPart->isLoaded()) {
					loadAgentsPartition(aPart, d);
				}
			}
		}

		// execute the call on the partition
		//Place** devPlaces = d.devPlaces[ placeHandle ]; // put this pointer in the partition
		//callAllPlacesKernel <<<pPart->blockDim ( ), pPart->threadDim ( ), d.inputStream >>>( devPlaces, functionId, devArg, argSize);
		__cudaCheckError(__FILE__, __LINE__);

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

void Dispatcher::refreshAgents(int handle) {
	// TODO get the unload the slices for this handle from the GPU without deleting
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

void Dispatcher::loadPlacesPartition(PlacesPartition *part, DeviceConfig d) {
	cudaSetDevice(d.deviceNum);
	loadedPlaces[part] = d;

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
	cudaMemcpyAsync(dPtr, part->hostPtrPlusGhosts(), numBytes,
			cudaMemcpyHostToDevice, d.inputStream);

	// TODO set pointer array on GPU
}

void Dispatcher::getPlacesPartition(PlacesPartition *part,
		bool freeOnRetrieve) {
	DeviceConfig d = loadedPlaces[part];
	cudaSetDevice(d.deviceNum);

	// get partition onto device
	char* dPtr = (char*) part->devicePtr(); // again, use of char* to allow pointer arithmitic
	dPtr += part->getPlaceBytes() * part->getGhostWidth(); // we don't want to copy out bad ghost data
	int numBytes = part->getPlaceBytes() * part->size();
	cudaMemcpyAsync(part->hostPtr(), dPtr, numBytes, cudaMemcpyDeviceToHost,
			d.outputStream);

	if (freeOnRetrieve) {
		// update model state
		part->setDevicePtr(NULL);
		part->setLoaded(false);

		cudaFree(part->devicePtr());
		loadedPlaces.erase(part);
	}
}

void Dispatcher::loadAgentsPartition(AgentsPartition *part, DeviceConfig d) {
	cudaSetDevice(d.deviceNum);
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
	cudaMemcpyAsync(dPtr, part->hostPtrPlusGhosts(), numBytes,
			cudaMemcpyHostToDevice, d.inputStream);

	// TODO set pointer array on GPU
}

void Dispatcher::getAgentsPartition(AgentsPartition *part,
		bool freeOnRetrieve) {
	DeviceConfig d = loadedAgents[part];
	cudaSetDevice(d.deviceNum);

	// get partition onto device
	char* dPtr = (char*) part->devicePtr(); // again, use of char* to allow pointer arithmitic
	dPtr += part->getPlaceBytes() * part->getGhostWidth(); // we don't want to copy out bad ghost data
	int numBytes = part->getPlaceBytes() * part->size();
	cudaMemcpyAsync(part->hostPtr(), dPtr, numBytes, cudaMemcpyDeviceToHost,
			d.outputStream);

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

	// if possible, run simulation one partition per GPU
	// TODO phase II
	// else create n GPU-sized partitions
	// TODO phase III
}

void Dispatcher::configureAgents(Agents *agents) {
	// TODO implement
}

//int ngpu;                   // number of GPUs in use
//std::map<PlacesPartition *, DeviceConfig> loadedPlaces; // tracks which partition is loaded on which GPU
//std::map<AgentsPartition*, DeviceConfig> loadedAgents; // tracks whicn partition is loaded on which GPU
//std::map<int, DeviceConfig> deviceInfo;

}// namespace mass

