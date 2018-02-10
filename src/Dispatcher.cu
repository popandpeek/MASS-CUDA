
#include <sstream>
#include <algorithm>  // array compare
#include <iterator>

#include "Dispatcher.h"
#include "cudaUtil.h"
#include "Logger.h"

#include "DeviceConfig.h"
#include "Place.h"
#include "PlacesModel.h"
#include "Places.h"
#include "DataModel.h"

// caching for optimizing performance of setNeighborPlacesKernel():
__constant__ int offsets_device[MAX_NEIGHBORS]; 

using namespace std;

namespace mass {

__global__ void callAllPlacesKernel(Place **ptrs, int nptrs, int functionId,
		void *argPtr) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < nptrs) {
		ptrs[idx]->callMethod(functionId, argPtr);
	}
}

__global__ void callAllAgentsKernel(Agent **ptrs, int nptrs, int functionId,
        void *argPtr) {

    int idx = getGlobalIdx_1D_1D();

    if ((idx < nptrs) && (ptrs[idx] -> isAlive())) {
        ptrs[idx]->callMethod(functionId, argPtr);
    }
}

/**
 * neighbors is converted into a 1D offset of relative indexes before calling this function
 */
__global__ void setNeighborPlacesKernel(Place **ptrs, int nptrs, int nNeighbors) {
	int idx = getGlobalIdx_1D_1D();

    if (idx < nptrs) {
        PlaceState *state = ptrs[idx]->getState();

        for (int i = 0; i < nNeighbors; ++i) {
            int j = idx + offsets_device[i];
            if (j >= 0 && j < nptrs) {
                state->neighbors[i] = ptrs[j];
            } else {
                state->neighbors[i] = NULL;
            }
        }
    }
}

__global__ void setNeighborPlacesKernel(Place **ptrs, int nptrs, int nNeighbors, int functionId,
        void *argPtr) {
    int idx = getGlobalIdx_1D_1D();

    if (idx < nptrs) {
        PlaceState *state = ptrs[idx]->getState();

        for (int i = 0; i < nNeighbors; ++i) {
            int j = idx + offsets_device[i];
            if (j >= 0 && j < nptrs) {
                state->neighbors[i] = ptrs[j];
            } else {
                state->neighbors[i] = NULL;
            }
        }

        ptrs[idx]->callMethod(functionId, argPtr);
    }
}

// __global__  void terminateAgentsKernel(Agent **ptrs, int nptrs) {
//     int idx = getGlobalIdx_1D_1D();
//     if ((idx < nptrs) && (ptrs[idx] -> isAlive() == false)) {
//         // printf("TERMINATING AGENT %d\n", ptrs[idx] ->getIndex());

//         Place* place = ptrs[idx] -> getPlace();
//         place -> removeAgent(ptrs[idx]);

//         //TODO: update the total count of agents & release agent into free pool
//     }
// }

__global__ void resolveMigrationConflictsKernel(Place **ptrs, int nptrs) {
    int idx = getGlobalIdx_1D_1D();
    if (idx < nptrs) {
        ptrs[idx] -> resolveMigrationConflicts();
    }
}

__global__ void updateAgentLocationsKernel (Agent **ptrs, int nptrs) {
    int idx = getGlobalIdx_1D_1D();
    if (idx < nptrs) {
        Place* destination = ptrs[idx]->state->destPlace;
        if ( destination != NULL) {
            // check that the new Place is actually accepting the agent
            for (int i=0; i<MAX_AGENTS; i++) {
                if (destination->state->agents[i] == ptrs[idx]) {
                    // remove agent from the old place:
                    ptrs[idx] -> getPlace() -> removeAgent(ptrs[idx]);

                    // update place ptr in agent:
                    ptrs[idx] -> setPlace(destination);
                }
            }
            // clean all migration data:
            ptrs[idx]-> state->destPlace = NULL;
        }
    }
}

Dispatcher::Dispatcher() {
	model = NULL;
	initialized = false;
	neighborhood = NULL;
}

struct DeviceAndMajor {
	DeviceAndMajor(int device, int major) {
		this->device = device;
		this->major = major;
	}
	int device;
	int major;
};
bool compFunction (DeviceAndMajor i,DeviceAndMajor j) { return (i.major>j.major); }

void Dispatcher::init() {
	if (!initialized) {
		initialized = true;
		Logger::debug(("Initializing Dispatcher"));
		int gpuCount;
		cudaGetDeviceCount(&gpuCount);

		if (gpuCount == 0) {
			throw MassException("No GPU devices were found.");
		}

		vector<DeviceAndMajor> devices;
		for (int d = 0; d < gpuCount; d++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, d);

			Logger::debug("Device %d has compute capability %d.%d", d,
					deviceProp.major, deviceProp.minor);

			DeviceAndMajor deviceAndMajor = DeviceAndMajor(d, deviceProp.major);
			devices.push_back(deviceAndMajor);
		}

		//Sort devices by compute capability in descending order:
		std::sort (devices.begin(), devices.end(), compFunction);

		// Pick the device with the highest compute capability for simulation:
		deviceInfo = new DeviceConfig(devices[0].device);
		model = new DataModel();
	}
}

Dispatcher::~Dispatcher() {
	Logger::debug("Freeing deviceConfig");
	deviceInfo -> freeDevice();
}

// Updates the Places stored on CPU
Place** Dispatcher::refreshPlaces(int handle) {
    Logger::debug("Entering Dispatcher::refreshPlaces");
    PlacesModel *placesModel = model->getPlacesModel(handle);
	
    if (initialized) {
        Logger::debug("Dispatcher::refreshPlaces: Initialized -> copying info from GPU to CPU");

		void *devPtr = deviceInfo->getPlaceState(handle);

        int stateSize = placesModel->getStateSize();
		int qty = placesModel->getNumElements();
		int bytes = stateSize * qty;
		CATCH(cudaMemcpy(placesModel->getStatePtr(), devPtr, bytes, D2H));
	}

    Logger::debug("Exiting Dispatcher::refreshPlaces");
	return placesModel->getPlaceElements();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument, int argSize) {
	if (initialized) {
		Logger::debug("Dispatcher::callAllPlaces: Calling all on places[%d]. Function id = %d", placeHandle, functionId);

		// load any necessary arguments
		void *argPtr = NULL;
		if (argument != NULL) {
			deviceInfo->load(argPtr, argument, argSize);
		}

		Logger::debug("Dispatcher::callAllPlaces: Calling callAllPlacesKernel");
		PlacesModel *pModel = model->getPlacesModel(placeHandle);
		callAllPlacesKernel<<<pModel->blockDim(), pModel->threadDim()>>>(
				deviceInfo->getDevPlaces(placeHandle), pModel->getNumElements(),
				functionId, argPtr);
		CHECK();

		if (argPtr != NULL) {
			Logger::debug("Dispatcher::callAllPlaces: Freeing device args.");
			cudaFree(argPtr);
		}

		Logger::debug("Exiting Dispatcher::callAllPlaces()");
	}
}

// void *Dispatcher::callAllPlaces(int handle, int functionId, void *arguments[],
// 		int argSize, int retSize) {
// 	// perform call all
// 	callAllPlaces(handle, functionId, arguments, argSize);
// 	// get data from GPUs
// 	refreshPlaces(handle);
// 	// get necessary pointers and counts
// 	int qty = model->getPlacesModel(handle)->getNumElements();
// 	Place** places = model->getPlacesModel(handle)->getPlaceElements();
// 	void *retVal = malloc(qty * retSize);
// 	char *dest = (char*) retVal;

// 	for (int i = 0; i < qty; ++i) {
// 		// copy messages to a return array
// 		memcpy(dest, places[i]->getMessage(), retSize);
// 		dest += retSize;
// 	}
// 	return retVal;
// }

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
	Logger::debug("Inside Dispatcher::updateNeighborhood");

    neighborhood = vec;
    int nNeighbors = vec->size();
    Logger::debug("______new nNeighbors=%d", nNeighbors);

    int *offsets = new int[nNeighbors]; 

    PlacesModel *p = model->getPlacesModel(handle);
    int nDims = p->getNumDims();
    int *dimensions = p->getDims();

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
        Logger::debug("offsets[%d] = %d", j, offsets[j]); 
    }
    
    // Now copy offsets to the GPU:
    cudaMemcpyToSymbol(offsets_device, offsets, sizeof(int) * nNeighbors);
    CHECK();
    // cudaMemcpyToSymbol(nNeighbors_device, &nNeighbors, sizeof(int));
    // CHECK();

    delete [] offsets;
    Logger::debug("Exiting Dispatcher::updateNeighborhood");
    return true;
}

void Dispatcher::exchangeAllPlaces(int handle, std::vector<int*> *destinations) {
	Logger::debug("Inside Dispatcher::exchangeAllPlaces");
	
    if (destinations != neighborhood) {
        updateNeighborhood(handle, destinations);
    }

	Place** ptrs = deviceInfo->getDevPlaces(handle);
	int nptrs = deviceInfo->countDevPlaces(handle);
	PlacesModel *p = model->getPlacesModel(handle);

	Logger::debug("Launching Dispatcher::setNeighborPlacesKernel()");
	setNeighborPlacesKernel<<<p->blockDim(), p->threadDim()>>>(ptrs, nptrs, destinations -> size());
	CHECK();
	Logger::debug("Exiting Dispatcher::exchangeAllPlaces");
}

/* Collects data from neighbors and executes the 
 * specified functon on each of the places
 */
void Dispatcher::exchangeAllPlaces(int handle, std::vector<int*> *destinations, int functionId, 
        void *argument, int argSize) {
    Logger::debug("Inside Dispatcher::exchangeAllPlaces with functionId = %d as an argument", functionId);
    if (destinations != neighborhood) {
        updateNeighborhood(handle, destinations);
    }

    // load any necessary arguments
    void *argPtr = NULL;
    if (argument != NULL) {
        deviceInfo->load(argPtr, argument, argSize);
    }

    Place** ptrs = deviceInfo->getDevPlaces(handle);
    int nptrs = deviceInfo->countDevPlaces(handle);
    PlacesModel *p = model->getPlacesModel(handle);

    setNeighborPlacesKernel<<<p->blockDim(), p->threadDim()>>>(ptrs, nptrs, destinations -> size(), functionId, argPtr);
    CHECK();
    Logger::debug("Exiting Dispatcher::exchangeAllPlaces with functionId = %d as an argument", functionId);
}

Agent** Dispatcher::refreshAgents(int handle, int& numAgents) {
    //TODO: is numAgents updated?
    Logger::debug("Entering Dispatcher::refreshAgents");
    AgentsModel *agentsModel = model->getAgentsModel(handle);
    
    if (initialized) {
        Logger::debug("Dispatcher::refreshAgents: Initialized -> copying info from GPU to CPU");
        void *devPtr = deviceInfo->getAgentsState(handle);
        int stateSize = agentsModel->getStateSize();
        int qty = agentsModel->getNumElements(); //TODO: changing number of agents!!!
        int bytes = stateSize * qty;
        CATCH(cudaMemcpy(agentsModel->getStatePtr(), devPtr, bytes, D2H));
    }

    Logger::debug("Exiting Dispatcher::refreshAgents");
    return agentsModel->getAgentElements();
}

void Dispatcher::callAllAgents(int agentHandle, int functionId, void *argument,
            int argSize) {

    if (initialized) {
        Logger::debug("Dispatcher::callAllAgents: Calling all on agents[%d]. Function id = %d", agentHandle, functionId);

        // load any necessary arguments
        void *argPtr = NULL;
        if (argument != NULL) {
            deviceInfo->load(argPtr, argument, argSize);
        }

        Logger::debug("Dispatcher::callAllAgents: Calling callAllAgentsKernel");
        AgentsModel *aModel = model->getAgentsModel(agentHandle);
        callAllAgentsKernel<<<aModel->blockDim(), aModel->threadDim()>>>(
                deviceInfo->getDevAgents(agentHandle), aModel->getNumElements(),
                functionId, argPtr);
        CHECK();

        if (argPtr != NULL) {
            Logger::debug("Dispatcher::callAllAgents: Freeing device args.");
            cudaFree(argPtr);
        }

        Logger::debug("Exiting Dispatcher::callAllAgents()");
    }
}

void Dispatcher::terminateAgents(int agentHandle) {
    // Logger::debug("Inside Dispatcher::terminateAgents");

    // AgentsModel *aModel = model->getAgentsModel(agentHandle);

    // terminateAgentsKernel<<<aModel->blockDim(), aModel->threadDim()>>>(
    //         deviceInfo->getDevAgents(agentHandle), aModel->getNumElements());
    // CHECK();

    // Logger::debug("Exiting Dispatcher::terminateAgents");

}

void Dispatcher::migrateAgents(int agentHandle, int placeHandle) {
    // Resolve migration colflicts between agents
    Place** p_ptrs = deviceInfo->getDevPlaces(placeHandle);
    PlacesModel *p = model->getPlacesModel(placeHandle);

    // printf("\nLaunching Dispatcher::resolveMigrationConflictsKernel()\n");
    resolveMigrationConflictsKernel<<<p->blockDim(), p->threadDim()>>>(p_ptrs, p->getNumElements());
    CHECK();


    AgentsModel *a = model->getAgentsModel(agentHandle);
    Agent **a_ptrs = deviceInfo->getDevAgents(agentHandle);

    Logger::debug("Launching Dispatcher:: updateAgentLocationsKernel()");
    updateAgentLocationsKernel<<<a->blockDim(), a->threadDim()>>>(a_ptrs, a->getNumElements());
    CHECK();
}

}// namespace mass

