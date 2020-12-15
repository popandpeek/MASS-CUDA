
#include <sstream>
#include <algorithm>  // array compare
#include <iterator>
#include <typeinfo>

#include "Dispatcher.h"
#include "cudaUtil.h"
#include "Logger.h"

#include "DeviceConfig.h"
#include "Place.h"
#include "PlacesModel.h"
#include "Places.h"
#include "DataModel.h"

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
            // TODO: Doesn't this result in empty spaces in array == to nChildren?
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
        int gpuCount;
        cudaGetDeviceCount(&gpuCount);
        
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
                    Logger::debug("Device[%d] linked with Device[%d].", devices.at(i), devices.at(j));
                }
            }
        }

        deviceInfo = new DeviceConfig(devices);
        model = new DataModel(devices.size());
	}
}

Dispatcher::~Dispatcher() {
	deviceInfo->freeDevice();
}

// Updates the Places stored on CPU
std::vector<Place**> Dispatcher::refreshPlaces(int handle) {
    Logger::debug("Entering Dispatcher::refreshPlaces");
    PlacesModel *placesModel = model->getPlacesModel(handle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(handle);
    int* dims = deviceInfo->getDimSize();
    std::vector<int> devices = deviceInfo->getDevices();
    if (initialized) {
        Logger::debug("Dispatcher::refreshPlaces: Initialized -> copying info from GPU to CPU");
        std::vector<void*> devPtrs = deviceInfo->getPlaceStates(handle);
        int placesStrideDev = deviceInfo->getPlacesStride(handle);
        int stateSize = placesModel->getStateSize();
        Logger::debug("Dispatcher::refreshPlaces: devPtrs size = %d; placesStrideDev = %d; stateSize = %d", devPtrs.size(), placesStrideDev, stateSize);
        for (int i = 0; i < devices.size(); ++i) {
            Logger::debug("Dispatcher::refreshPlaces: copy memory on device: %d", i);
            int bytes = stateSize * (placesStrideDev + dims[1] * ghostPlaceMult[i]);
            cudaSetDevice(devices.at(i));
            CATCH(cudaMemcpy(placesModel->getStatePtr(i), devPtrs.at(i), bytes, D2H));
        }
	}
    
    Logger::debug("Exiting Dispatcher::refreshPlaces");
    return placesModel->getPlaceElements();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument, int argSize) {
	if (initialized) {
        Logger::debug("Dispatcher::callAllPlaces: Calling all on places[%d]. Function id = %d", 
                placeHandle, functionId);

        std::vector<int> devices = deviceInfo->getDevices();
        std::vector<Place**> devPtrs = deviceInfo->getDevPlaces(placeHandle); 
        int stride = deviceInfo->getPlacesStride(placeHandle);
        dim3* pDims = deviceInfo->getPlacesThreadBlockDims(placeHandle);
        int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(placeHandle);
        int* dims = deviceInfo->getDimSize();
        void *argPtr = NULL;
        for (int i = 0; i < devices.size(); ++i) {
            Logger::debug("Launching Dispatcher::callAllPlacesKernel() on device: %d with params: placesStride == %d and ghostPlaceSize = %d", 
                    devices.at(i), stride, ghostPlaceMult[i] * dims[0]);
            cudaSetDevice(devices.at(i));

            // load any necessary arguments
            if (argument != NULL) {
                CATCH(cudaMalloc((void** ) &argPtr, argSize));
                CATCH(cudaMemcpy(argPtr, argument, argSize, H2D));
            }
            // pass params to kernel function to allow for ghost places processing
            callAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(i), stride + ghostPlaceMult[i] * dims[0], functionId, argPtr);
            CHECK();
            cudaDeviceSynchronize();
            if (argPtr != NULL) {
                Logger::debug("Dispatcher::callAllPlaces: Freeing device args.");
                cudaFree(argPtr);
            }
        }

		Logger::debug("Exiting Dispatcher::callAllPlaces()");
	}
}

bool Dispatcher::updateNeighborhood(int handle, std::vector<int*> *vec) {
	Logger::debug("Inside Dispatcher::updateNeighborhood");

    neighborhood = vec;
    int nNeighbors = vec->size();
    Logger::debug("______new nNeighbors=%d", nNeighbors);

    int *offsets = new int[nNeighbors]; 

    int nDims = deviceInfo->getDimensions();
    int *dimensions = deviceInfo->getDimSize();

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
    for (int i = 0; i < deviceInfo->getNumDevices(); ++i) {
        CATCH(cudaSetDevice(deviceInfo->getDeviceNum(i)));
        cudaMemcpyToSymbol(offsets_device, offsets, sizeof(int) * nNeighbors);
        CHECK();
        Logger::debug("Copied constant memory to device %d", i);
    }
    cudaDeviceSynchronize();

    delete [] offsets;
    Logger::debug("Exiting Dispatcher::updateNeighborhood");
    return true;
} 

void Dispatcher::exchangeAllPlaces(int handle, std::vector<int*> *destinations) {
	Logger::debug("Inside Dispatcher::exchangeAllPlaces");
	
    if (destinations != neighborhood) {
        updateNeighborhood(handle, destinations);
    }

	std::vector<Place**> devPtrs = deviceInfo->getDevPlaces(handle);

    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(handle);
    Logger::debug("Kernel dims = gridDim { %d, %d, %d } and blockDim = { %d, %d, %d }", pDims[0].x, pDims[0].y, pDims[0].z, pDims[1].x, pDims[1].y, pDims[1].z);
    std::vector<int> devices = deviceInfo->getDevices();
    int stride = deviceInfo->getPlacesStride(handle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(handle);
    int* dims = deviceInfo->getDimSize();
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        // TODO: Need to pass params to kernel function to allow for ghost places processing
        Logger::debug("DispatcherExchangeAllPlaces: nptrs = %d", stride + (dims[0] * ghostPlaceMult[i]));
        exchangeAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(i), stride + (dims[0] * ghostPlaceMult[i]), destinations->size());
        CHECK();
        cudaDeviceSynchronize();
    }

	Logger::debug("Exiting Dispatcher::exchangeAllPlaces");
}

std::vector<Agent**> Dispatcher::refreshAgents(int handle) {
    Logger::debug("Entering Dispatcher::refreshAgents");
    AgentsModel *agentsModel = model->getAgentsModel(handle);
    std::vector<int> devices = deviceInfo->getDevices();

    if (initialized) {
        Logger::debug("Dispatcher::refreshAgents: Initialized -> copying info from GPU to CPU");
        std::vector<void*> devPtrs = deviceInfo->getAgentsState(handle);
        int* agentsPerDevice = deviceInfo->getnAgentsDev(handle);

        int stateSize = agentsModel->getStateSize();
        Logger::debug("Dispatcher::refreshAgents: devPtrs size = %d; agentsPerDevice = %d; stateSize = %d", devPtrs.size(), agentsPerDevice[0], stateSize);

        for (int i = 0; i < devices.size(); ++i) {
            Logger::debug("Dispatcher::refreshAgents: copy memory on device: %d", i);
            cudaSetDevice(devices[i]);
            CATCH(cudaMemcpy(agentsModel->getStatePtr(i), devPtrs.at(i), stateSize * agentsPerDevice[i], D2H));
        }
    }

    Logger::debug("Exiting Dispatcher::refreshAgents");
    return agentsModel->getAgentElements();
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

    std::vector<Place**> devPtrs = deviceInfo->getDevPlaces(handle);
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(handle);
    std::vector<int> devices = deviceInfo->getDevices();
    int stride = deviceInfo->getPlacesStride(handle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(handle);
    int* dims = deviceInfo->getDimSize();
    Logger::debug("Kernel dims = gridDim { %d, %d, %d } and blockDim = { %d, %d, %d }", pDims[0].x, pDims[0].y, pDims[0].z, pDims[1].x, pDims[1].y, pDims[1].z);
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        // load any necessary arguments
        void *argPtr = NULL;
        if (argument != NULL) {
            CATCH(cudaMalloc((void** ) &argPtr, argSize));
		    CATCH(cudaMemcpy(argPtr, argument, argSize, H2D));
        }
        Logger::debug("DispatcherExchangeAllPlaces: nptrs = %d", stride + (dims[0] * ghostPlaceMult[i]));
        // TODO: Need to pass params to kernel function to allow for ghost places processing
        exchangeAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(i), stride + (dims[0] * ghostPlaceMult[i]), destinations->size(), functionId, argPtr);
        CHECK();
        cudaDeviceSynchronize();
        if (argPtr != NULL) {
            Logger::debug("Dispatcher::exchangeAllPlaces: Freeing device args.");
            cudaFree(argPtr);
        }
    }

    Logger::debug("Exiting Dispatcher::exchangeAllPlaces with functionId = %d as an argument", functionId);
} 

void Dispatcher::callAllAgents(int agentHandle, int functionId, void *argument, int argSize) {
    if (initialized) {
        Logger::debug("Dispatcher::callAllAgents: Calling all on agents[%d]. Function id = %d", agentHandle, functionId);

        dim3* aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);
        Logger::debug("Kernel dims = gridDim %d and blockDim = %d", aDims[0].x, aDims[1].x);

        std::vector<int> devices = deviceInfo->getDevices();
        std::vector<Agent**> agtsPtrs = deviceInfo->getDevAgents(agentHandle); 
        int* strides = deviceInfo->getnAgentsDev(agentHandle);

        for (int i = 0; i < devices.size(); ++i) {
            
            Logger::debug("Launching Dispatcher::callAllAgentsKernel() on device: %d", devices.at(i));
            cudaSetDevice(devices.at(i));
            // load any necessary arguments
            void *argPtr = NULL;
            if (argument != NULL) {
                CATCH(cudaMalloc((void** ) &argPtr, argSize));
			    CATCH(cudaMemcpy(argPtr, argument, argSize, H2D));
            }
            callAllAgentsKernel<<<aDims[0], aDims[1]>>>(agtsPtrs.at(i), strides[i], functionId, argPtr);
            CHECK();
            if (argPtr != NULL) {
                Logger::debug("Dispatcher::callAllAgents: Freeing device args.");
                cudaFree(argPtr);
            }
        }

        Logger::debug("Exiting Dispatcher::callAllAgents()");
    }
}

void Dispatcher::terminateAgents(int agentHandle) {
    //TODO: implement periodic garbage collection of terminated agents and reuse of that space to allocate new agents
    
    // 1. get agentHandle's Agent DS and bag of Agent DS
    std::vector<int> devices = deviceInfo->getDevices();
    std::vector<Agent**> agentDevPtrs = deviceInfo->getDevAgents(agentHandle);
    std::vector<Agent**> bagOAgentsDevPtrs = deviceInfo->getBagOAgentsDevPtrs(agentHandle);
    int* nAgentsDev = deviceInfo->getnAgentsDev(agentHandle);

    // 2. int *var to track number of agents to be collected 
    std::vector<int*> hAgentTerminateCount;
    std::vector<int*> dAgentTerminateCount;
    for (int i = 0; i < devices.size(); ++i) {
        int tmp = 0;
        int* tptr = NULL;
        hAgentTerminateCount.push_back(&tmp);
        dAgentTerminateCount.push_back(tptr);
    }

    // 2. iterate over devices
    //     a. activate device
    //     b. kernel function to accumulate count of dead agents from each place
    dim3* aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);
    for (int i = 0; i < agentDevPtrs.size(); ++i) {
        cudaSetDevice(devices.at(i));
        cudaMalloc((void** ) &(dAgentTerminateCount.at(i)), sizeof(int));
        cudaMemcpy(dAgentTerminateCount.at(i), hAgentTerminateCount.at(i), sizeof(int), H2D);
        // terminateAgentsCount<<<aDims[0], aDims[1]>>>(agentDevPtrs.at(i), nAgentsDev[i], dAgentTerminateCount.at(i));
    }

    // 3. initialize ds for each bag of collected agents
    // 4. iterate over devices
    //     a. activate device
    //     b. kernel function to add dead agents to each devices (SHARED?) agent bag
}   

void Dispatcher::migrateAgents(int agentHandle, int placeHandle) {
    Logger::debug("Inside Dispatcher:: migrateAgents().");
    std::vector<Place**> p_ptrs = deviceInfo->getDevPlaces(placeHandle);
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(placeHandle);
    Logger::debug("resolveMigrationConflicts Kernel dims = gridDim %d and blockDim = %d", pDims[0].x, pDims[1].x);
    std::vector<int> devices = deviceInfo->getDevices();
    int placeStride = deviceInfo->getPlacesStride(placeHandle);
    Logger::debug("Dispatcher::MigrateAgents: number of places: %d", deviceInfo->getPlaceCount(placeHandle));
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher:: resolveMigrationConflictsKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        resolveMigrationConflictsKernel<<<pDims[0], pDims[1]>>>(p_ptrs.at(i), placeStride);
        CHECK();		
    }

    std::vector<Agent**> a_ptrs = deviceInfo->getDevAgents(agentHandle);
    dim3* aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);
    int* agentStrides = deviceInfo->getnAgentsDev(agentHandle);    
    Logger::debug("Dispatcher::MigrateAgents: number of agents: %d", deviceInfo->getNumAgents(agentHandle));
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher:: updateAgentLocationsKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        updateAgentLocationsKernel<<<aDims[0], aDims[1]>>>(a_ptrs.at(i), agentStrides[i]);
        CHECK();
    }

    Logger::debug("Exiting Dispatcher:: migrateAgents().");
}

void Dispatcher::spawnAgents(int handle) {
    
    Logger::debug("Inside Dispatcher::spawnAgents()");
    std::vector<Agent**> a_ptrs = deviceInfo->getDevAgents(handle);
    dim3* aDims = deviceInfo->getAgentsThreadBlockDims(handle);
    Logger::debug("Kernel dims = gridDim %d and blockDim = %d", aDims[0].x, aDims[1].x);

    std::vector<int> devices = deviceInfo->getDevices();
    int* nAgentsDevs = deviceInfo->getnAgentsDev(handle);

    int* numAgentObjects[devices.size()];
    for (int i = 0; i < devices.size(); ++i) { 
        cudaSetDevice(devices.at(i));
        CATCH(cudaMalloc(&numAgentObjects[i], sizeof(int)));
        CATCH(cudaMemcpy(numAgentObjects[i], &nAgentsDevs[i], sizeof(int), H2D));
    }

    for (int i = 0; i < devices.size(); ++i) {
        cudaSetDevice(devices.at(i));
        spawnAgentsKernel<<<aDims[0], aDims[1]>>>(a_ptrs.at(i), numAgentObjects[i], deviceInfo->getMaxAgents(handle, i));
        CHECK();

        // Is this necessary? If so, may need to accumulate count of results above
        if (*numAgentObjects[i] > deviceInfo->getMaxAgents(handle, i)) {
            throw MassException("Trying to spawn more agents than the maximun set for the system");
        }

        int nNewAgents = *numAgentObjects[i] - getNumAgentsInstantiated(handle)[i];
        deviceInfo->devAgentsMap[handle].nAgents += nNewAgents;
        deviceInfo->devAgentsMap[handle].nAgentsDev[i] += nNewAgents;
    }

    Logger::debug("Finished Dispatcher::spawnAgents");
}

int Dispatcher::getNumAgents(int agentHandle) {
    return deviceInfo->getNumAgents(agentHandle);
}

int* Dispatcher::getMaxAgents(int agentHandle) {
    return deviceInfo->getMaxAgents(agentHandle);
}

int* Dispatcher::getNAgentsDev(int handle) {
    return deviceInfo->getnAgentsDev(handle);
}

int Dispatcher::getNumPlaces(int handle) {
    return deviceInfo->getPlaceCount(handle);
}

int Dispatcher::getPlacesStride(int handle) {
    return (deviceInfo->getPlaceCount(handle) / deviceInfo->getNumDevices());
}

int* Dispatcher::getGhostPlaceMultiples(int handle) {
    return deviceInfo->getGhostPlaceMultiples(handle);
}

int* Dispatcher::getNumAgentsInstantiated(int handle) {
    return deviceInfo->getMaxAgents(handle);
}

}// namespace mass

