
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
// #include "DataModel.h"

// using constant memory to optimize the performance of exchangeAllPlacesKernel():
__constant__ int offsets_device[MAX_NEIGHBORS]; 

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
__global__ void exchangeAllPlacesKernel(Place **ptrs, int nptrs, int nNeighbors) {
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

__global__ void exchangeAllPlacesKernel(Place **ptrs, int nptrs, int nNeighbors, int functionId,
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

__global__ void spawnAgentsKernel(Agent **ptrs, int* nextIdx, int maxAgents) {
    int idx = getGlobalIdx_1D_1D();
    if (idx < *nextIdx) {
        if ((ptrs[idx]->isAlive()) && (ptrs[idx]->state->nChildren > 0)) {
            // find a spot in Agents array:
            int idxStart = atomicAdd(nextIdx, ptrs[idx]->state->nChildren);
            if (idxStart+ptrs[idx]->state->nChildren >= maxAgents) {
                return;
            }
            for (int i=0; i< ptrs[idx]->state->nChildren; i++) {
                // instantiate with proper index
                ptrs[idxStart+i]->setAlive();
                ptrs[idxStart+i]->setIndex(idxStart+i);

                // link to a place:
                ptrs[idxStart+i] -> setPlace(ptrs[idx]->state->childPlace);
                ptrs[idx]->state->childPlace -> addAgent(ptrs[idxStart+i]);
            }

            // restore Agent spawning data:
            ptrs[idx]->state->nChildren = 0;
            ptrs[idx]->state->childPlace = NULL;

        }
    }
}

Dispatcher::Dispatcher() {
	initialized = false;
	neighborhood = NULL;
}

void Dispatcher::init() {
	if (!initialized) {
		initialized = true; 
		Logger::debug(("Initializing Dispatcher"));

		if (gpuCount == 0) {
			throw MassException("No GPU devices were found.");
		}

        // Establish peerable device list
        std::vector<int> devices;
        // CUDA runtime places highest CC device in first position, no further ordering guaranteed
        devices.push_back(0);
        for (int d = 1; d < gpuCount; d++) {
            int canAccessPeer = 0;
            // checks that each device can peer with first 
            CATCH(cudaDeviceCanAccessPeer(&canAccessPeer, 0, d));
            if (canAccessPeer) {
                devices.push_back(d);
            }
        }

        // Establish bi-directional peer relationships for all peerable devices
        for (std::size_t i = 0; i < devices.size(); ++i) {
            cudaSetDevice(devices.at(i));
            for (std::size_t j = 0; j < devices.size(); ++j) {
                if (i != j) {
                    CATCH(cudaDeviceEnablePeerAccess(devices.at(j), 0));
                }
            }
        }

        deviceInfo = new DeviceConfig(devices);
	}
}

Dispatcher::~Dispatcher() {
	deviceInfo->freeDevice();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument, int argSize) {
	if (initialized) {
        Logger::debug("Dispatcher::callAllPlaces: Calling all on places[%d]. Function id = %d", 
                placeHandle, functionId);

		// load any necessary arguments
		void *argPtr = NULL;
		if (argument != NULL) {
			deviceInfo->load(argPtr, argument, argSize);
		}

		Logger::debug("Dispatcher::callAllPlaces: Calling callAllPlacesKernel");
		dim3* dims = deviceInfo->getThreadBlockDims();

        std::vector<int> devices = deviceInfo->getDevices();
        int numPlaces = deviceInfo->countDevPlaces(placeHandle);
        Place** devPtr = deviceInfo->getDevPlaces(placeHandle);

        // TODO: need to handle cases where places do not split even
        int stride = numPlaces / devices.size();
        for (int i = 0; i < devices.size(); ++i) {
            void *tempArgPtr = NULL;
            if (argPtr != NULL) {
                tempArgPtr = argPtr + i * stride;
            }

            Logger::debug("Launching Dispatcher::callAllPlacesKernel() on device: %d", devices.at(i));
            CATCH(cudaSetDevice(devices.at(i)));
            callAllPlacesKernel<<<dims[0], dims[1]>>>(devPtr + i * stride, stride, 
                    functionId, tempArgPtr);
		    CHECK();
        }

		if (argPtr != NULL) {
			Logger::debug("Dispatcher::callAllPlaces: Freeing device args.");
			CATCH(cudaFree(argPtr));
		}

		Logger::debug("Exiting Dispatcher::callAllPlaces()");
	}
}

bool Dispatcher::updateNeighborhood(int handle, vector<int*> *vec) {
	Logger::debug("Inside Dispatcher::updateNeighborhood");

    neighborhood = vec;
    int nNeighbors = vec->size();
    Logger::debug("______new nNeighbors=%d", nNeighbors);

    int *offsets = new int[nNeighbors]; 

    int nDims = deviceInfo->getDims();
    int *dimensions = deviceInfo->getSize();

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
    dim3* dims = deviceInfo->getBlockThreadDims(handle);
    std::vector<int> devices = deviceInfo->getDevices();
    int stride = numPlaces / devices.size();

    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", devices.at(i));
        CATCH(cudaSetDevice(devices.at(i)));
        exchangeAllPlacesKernel<<<dims[0], dims[1]>>>(ptrs + i * stride, stride, 
                destinations->size());
        CHECK();
    }

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
    dim3* dims = deviceInfo->getBlockThreadDims(handle);
    std::vector<int> devices = deviceInfo->getDevices();
    int stride = numPlaces / devices.size();

    for (int i = 0; i < devices.size(); ++i) {
        void *tempArgPtr = NULL;
        if (argPtr != NULL) {
            tempArgPtr = argPtr + i * stride;
        }

        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", devices.at(i));
        CATCH(cudaSetDevice(devices.at(i)));
        exchangeAllPlacesKernel<<<dims[0], dims[1]>>>(ptrs + i * stride, stride, 
                destinations->size(), functionId, tempArgPtr);
        CHECK();
    }

    Logger::debug("Exiting Dispatcher::exchangeAllPlaces with functionId = %d as an argument", functionId);
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

        dim3* dims = deviceInfo->getDims(agentHandle);

        std::vector<int> devices = deviceInfo->getDevices();
        Agent** agtsPtr = deviceInfo->getDevAgents(agentHandle);
        int numAgentObjects = deviceInfo->getNumAgentObjects(agentHandle);
        int stride = numAgentObjects / devices.size();
        for (int i = 0; i < devices.size(); ++i) {
            void *tempArgPtr = NULL;
            if (argPtr != NULL) {
                tempArgPtr = argPtr + i * stride * sizeof(void*);
            }

            Logger::debug("Launching Dispatcher::callAllAgentsKernel() on device: %d", devices.at(i));
            CATCH(cudaSetDevice(devices.at(i)));
            callAllAgentsKernel<<<dims[0], dims[1]>>>(agtsPtr + i * stride, stride, 
                    functionId, tempArgPtr);
            CHECK();
        }
        if (argPtr != NULL) {
            Logger::debug("Dispatcher::callAllAgents: Freeing device args.");
            cudaFree(argPtr);
        }

        Logger::debug("Exiting Dispatcher::callAllAgents()");
    }
}

void Dispatcher::terminateAgents(int agentHandle) {
    //TODO: implement periodic garbage collection of terminated agents and reuse of that space to allocate new agents
}

void Dispatcher::migrateAgents(int agentHandle, int placeHandle) {
    Place** p_ptrs = deviceInfo->getDevPlaces(placeHandle);
    int numPlaces = deviceInfo->countDevPlaces(place));
    dim3* dims = deviceInfo->getBlockThreadDims(agentHandle);
    st::vector<int> devices = deviceInfo->getDevices();
    int placeStride = numPlaces / devices.size();

    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher:: resolveMigrationConflictsKernel() on device: %d", 
                devices.at(i));
        CATCH(cudaSetDevice(devices.at(i)));
        resolveMigrationConflictsKernel<<<dims[0], dims[1]>>>(p_ptrs + i * placeStride, placeStride);
        CHECK();
    }

    Agent **a_ptrs = deviceInfo->getDevAgents(agentHandle);
    int numAgts = getNumAgentObjects(agentHandle);
    int agentStride = numAgts / devices.size();    

    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher:: updateAgentLocationsKernel() on device: %d", devices.at(i));
        CATCH(cudaSetDevice(devices.at(i)));
        updateAgentLocationsKernel<<<dims[0], dims[1]>>>(a_ptrs + i * agentStride, agentStride);
        CHECK();
    }
}

void Dispatcher::spawnAgents(int handle) {
    
    Logger::debug("Inside Dispatcher::spawnAgents");
    Agent **a_ptrs = deviceInfo->getDevAgents(handle);
    dim3* dims = deviceInfo->getBlockThreadDims(handle);

    //allocate numAgentObjects on in managed memory on GPU:
    int* numAgentObjects = new int(getNumAgentObjects(handle));
    CATCH(cudaMallocManaged(&numAgentObjects, sizeof(int)));
        
    // TODO: Loop over devices and call kernel function
    //       Need to implement even splitting algo and push all stride calculations to instantiation
    std::vector<int> devices = deviceInfo->getDevices();
    int numAgents = getNumAgents(handle);
    int stride = numAgents / devices.size();
    int maxAgents = deviceInfo->getMaxAgents(handle);
    for (int i = 0; i < devices.size(); ++i) {
        CATCH(cudaSetDevice(devices.at(i)));
        spawnAgentsKernel<<<dims[0], dims[1]>>>(a_ptrs + i * stride, numAgentObjects / stride, 
                maxAgents / stride);
        CHECK();
    }

    // Is this necessary? If so, may need to accumulate count of results above
    if (*numAgentObjects > deviceInfo->getMaxAgents(handle)) {
        throw MassException("Trying to spawn more agents than the maximun set for the system");
    }

    int nNewAgents = *numAgentObjects - getNumAgentObjects(handle);
    deviceInfo->devAgentsMap[handle].nAgents += nNewAgents;
    deviceInfo->devAgentsMap[handle].nextIdx += nNewAgents;
    Logger::debug("Finished Dispatcher::spawnAgents");
}

int Dispatcher::getNumAgents(int agentHandle) {
    return deviceInfo->getNumAgents(agentHandle);
}

int Dispatcher::getNumAgentObjects(int agentHandle) {
    return deviceInfo->getNumAgentObjects(agentHandle);
}

}// namespace mass

