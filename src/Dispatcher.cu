
#include <sstream>
#include <algorithm>  // array compare
#include <iterator>
#include <typeinfo>
#include "../cub-1.8.0/cub/cub.cuh"

#include "Dispatcher.h"
#include "cudaUtil.h"
#include "settings.h"
#include "Logger.h"

#include "DeviceConfig.h"
#include "Place.h"
#include "PlacesModel.h"
#include "Places.h"
#include "DataModel.h"

// using constant memory to optimize the performance of exchangeAllPlacesKernel():
__constant__ int offsets_device[MAX_NEIGHBORS]; 

namespace mass {

__global__ void callAllPlacesKernel(Place **ptrs, int nptrs, int functionId, int idxBump, void *argPtr) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < nptrs) {
		ptrs[idx + idxBump]->callMethod(functionId, argPtr);
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
__global__ void exchangeAllPlacesKernel(Place **ptrs, int nptrs, int idxStart, int idxEnd, int nNeighbors) {
	int idx = getGlobalIdx_1D_1D();

    if (idx < nptrs) {
        PlaceState *state = ptrs[idx + idxStart]->getState();

        for (int i = 0; i < nNeighbors; ++i) {
            int j = idx + idxStart + offsets_device[i];
            if (j >= 0 && j < nptrs + idxStart + idxEnd) {
                state->neighbors[i] = ptrs[j];
            } else {
                state->neighbors[i] = NULL;
            }
        }
    }
}

__global__ void exchangeAllPlacesKernel(Place **ptrs, int nptrs, int idxStart, int idxEnd, int nNeighbors, int functionId,
        void *argPtr) {
    unsigned idx = getGlobalIdx_1D_1D();

    if (idx < nptrs) {
        PlaceState *state = ptrs[idx + idxStart]->getState();

        for (int i = 0; i < nNeighbors; ++i) {
            int j = idx + idxStart + offsets_device[i];
            if (j >= 0 && j < nptrs + idxStart + idxEnd) {
                state->neighbors[i] = ptrs[j];
            } else {
                state->neighbors[i] = NULL;
            }
        }

        ptrs[idx + idxStart]->callMethod(functionId, argPtr);
    }
}

__global__ void setFlagsKernel(Agent** a_ptrs, int nPtrs, bool* flags) {
    unsigned idx = getGlobalIdx_1D_1D();
    if (idx < nPtrs) {
        flags[idx] = 0;
        if (a_ptrs[idx]->isAlive()) {
            flags[idx] = 1;
        }
    }
}

__global__ void writeMovingAgentLocationsKernel(AgentState* state_ptrs, int startIndex, unsigned qty, unsigned* locations, unsigned* locations_loc) {
    unsigned idx = getGlobalIdx_1D_1D() + startIndex;
    if (idx < qty) {
        if(state_ptrs[idx].isAlive) {
            int location_idx = atomicAdd(locations_loc, 1);
            locations[location_idx] = idx;
        }
    }
}

__global__ void compactAgentsKernel(AgentState* state_ptrs, int qty, int agentStateSize, unsigned* locations, unsigned* locations_loc) {
    unsigned idx = getGlobalIdx_1D_1D();
    if (idx < qty) {
        if (!(state_ptrs[idx].isAlive)) {
            int location_idx = atomicAdd(locations_loc, 1);
            AgentState temp = state_ptrs[idx];
            memcpy(&(state_ptrs[idx]), &(state_ptrs[locations[location_idx]]), agentStateSize);
            memcpy(&(state_ptrs[locations[location_idx]]), &(temp), agentStateSize);
        }
    }
}

__global__ void realignAgentsKernel(Agent** a_ptrs, AgentState* ste_ptrs, int nPtrs) {
    unsigned idx = getGlobalIdx_1D_1D();
    if (idx < nPtrs) {
        a_ptrs[idx]->state = &ste_ptrs[idx];
        a_ptrs[idx]->setIndex(idx);
    }
}

__global__ void unattachPlaceAgentsKernel(Place** p_ptrs, int placesStride, int ghostplace_offset) {
    unsigned idx = getGlobalIdx_1D_1D();
    if (idx < placesStride) {
        Place* pl = p_ptrs[idx + ghostplace_offset];
        for (int i = 0; i < MAX_AGENTS; i++) {
            if (pl->state->agents[i] != NULL) {
                pl->removeAgent(pl->state->agents[i]);
            }
        }
    }
}

__global__ void reattachPlaceAgentsKernel(Agent** a_ptrs, Place** p_ptrs, int nPtrs) {
    unsigned idx = getGlobalIdx_1D_1D();
    if (idx < nPtrs) {
        int pl_idx = a_ptrs[idx]->getPlaceDevIndex();
        Place* pl = p_ptrs[pl_idx];
        pl->reattachAgent(a_ptrs[idx]);
    }
}

__global__ void longDistanceMigrationKernel(Agent **src_agent_ptrs, Agent **dest_agent_ptrs, 
        AgentState *src_agent_state, AgentState *dest_agent_state, int nAgentsDevSrc, 
        int* nAgentsDevDest, int destDevice, int placesStride, int stateSize) {
    
    int idx = getGlobalIdx_1D_1D();
    if (idx < nAgentsDevSrc) {
        if (src_agent_ptrs[idx]->isAlive() && src_agent_ptrs[idx]->longDistanceMigration()) {
            unsigned destPlaceIdx = src_agent_state[idx].destPlaceIdx;
            if ((destPlaceIdx >= (placesStride * destDevice)) && 
                    (destPlaceIdx < (placesStride * destDevice + placesStride))) {
                int neighborIdx = atomicAdd(nAgentsDevDest, 1);
                memcpy(&(dest_agent_state[neighborIdx]), &(src_agent_state[idx]), stateSize);

                // clean up Agent in source array
        	    src_agent_ptrs[idx]->terminateAgent();
            }
        } 
    }
}

__global__ void longDistanceMigrationsSetPlaceKernel(Place** p_ptrs, Agent** a_ptrs, int qty, int placeStride,
        int ghostPlaces, int ghostSpaceMult, int device) {
    
    int idx = getGlobalIdx_1D_1D();
    if (idx < qty) {
        if (a_ptrs[idx]->isAlive() && a_ptrs[idx]->longDistanceMigration()) {
            a_ptrs[idx]->setLongDistanceMigration(false);
            int placePtrIdx = a_ptrs[idx]->getPlaceIndex() - (device * placeStride) + 
                    ((ghostPlaces * 2) - (ghostSpaceMult * ghostPlaces));
            if (p_ptrs[placePtrIdx]->addAgent(a_ptrs[idx])) {
                a_ptrs[idx]->setPlace(p_ptrs[placePtrIdx]);
                return;
            }
            // No home found on device traveled to so Agent is terminated on new device
            a_ptrs[idx]->terminateGhostAgent();
       }
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

__global__ void moveAgentsDownKernel(Agent **src_agent_ptrs, Agent **dest_agent_ptrs, 
            AgentState *src_agent_state, AgentState *dest_agent_state, 
            Place **src_place_ptrs, Place **dest_place_ptrs, 
            int device, int placesStride, int ghostPlaces, 
            int ghostPlaceMult, int nAgentsDevSrc, int *nAgentsDevDest, int stateSize) {

    int idx = getGlobalIdx_1D_1D();
    if (idx < nAgentsDevSrc) {
    // idx needs to be mapped base on which device L or R
        int place_index = src_agent_ptrs[idx]->getPlaceIndex();
        if (place_index >= (placesStride + (placesStride * device) + (ghostPlaceMult * ghostPlaces - ghostPlaces))) {
            int neighborIdx = atomicAdd(nAgentsDevDest, 1);
            memcpy(&(dest_agent_state[neighborIdx]), &(src_agent_state[idx]), stateSize);

            // clean up Agent in source array
        	src_agent_ptrs[idx]->terminateAgent();
		}
    }
}

__global__ void moveAgentsUpKernel(Agent **src_agent_ptrs, Agent **dest_agent_ptrs, 
            AgentState *src_agent_state, AgentState *dest_agent_state, 
            Place **src_place_ptrs, Place **dest_place_ptrs, 
            int device, int placesStride, int ghostPlaces, 
            int ghostPlaceMult, int nAgentsDevSrc, int *nAgentsDevDest, int stateSize) {

    int idx = getGlobalIdx_1D_1D();
    if (idx < nAgentsDevSrc) {
    // idx needs to be mapped base on which device L or R
        int place_index = src_agent_ptrs[idx]->getPlaceIndex();
        if (place_index < device * placesStride) {
            int neighborIdx = atomicAdd(nAgentsDevDest, 1);
			src_agent_ptrs[idx]->setTraveled(true);
            memcpy(&(dest_agent_state[neighborIdx]), (&(src_agent_state[idx])), stateSize);
            
            // clean up Agent in source array
			src_agent_ptrs[idx]->terminateAgent();
		}
    }
}

__global__ void updateAgentPointersMovingUp(Place** placePtrs, Agent** agentPtrs, 
		int qty, int placesStride, int ghostPlaces, int ghostSpaceMult, int device) {
	int idx = getGlobalIdx_1D_1D();
	if (idx < qty) {
		if (agentPtrs[idx]->isAlive() && agentPtrs[idx]->isTraveled()) {
			agentPtrs[idx]->setTraveled(false);
            // Get array index from overall indexing scheme
			int placePtrIdx = agentPtrs[idx]->getPlaceIndex() - (device * placesStride) + 
					(ghostPlaces + ghostPlaces * ghostSpaceMult);
			if (placePtrs[placePtrIdx]->addAgent(agentPtrs[idx])) {
				agentPtrs[idx]->setPlace(placePtrs[placePtrIdx]);
				return; 
			}
			// No home found on device traveled to so Agent is terminated on new device
			agentPtrs[idx]->terminateGhostAgent();
		}
	}
}

__global__ void updateAgentPointersMovingDown(Place** placePtrs, Agent** agentPtrs, 
		int qty, int placesStride, int ghostPlaces, int ghostSpaceMult, int device) {
	int idx = getGlobalIdx_1D_1D();
	if (idx < qty) {
		if (agentPtrs[idx]->isAlive() && agentPtrs[idx]->isTraveled()) {
			agentPtrs[idx]->setTraveled(false);
			int placePtrIdx = agentPtrs[idx]->getPlaceIndex() - (device * placesStride) + 
					((ghostPlaces * 2) - (ghostSpaceMult * ghostPlaces));
			if (placePtrs[placePtrIdx]->addAgent(agentPtrs[idx])) {
				agentPtrs[idx]->setPlace(placePtrs[placePtrIdx]);
				return; 
			}
			// No home found on device traveled to so Agent is terminated on new device
			agentPtrs[idx]->terminateGhostAgent();
		}
	}
}

__global__ void spawnAgentsKernel(Agent **ptrs, int* nextIdx, int maxAgents) {
    int idx = getGlobalIdx_1D_1D();
    if (idx < *nextIdx) {
        if ((ptrs[idx]->isAlive()) && (ptrs[idx]->state->nChildren  > 0)) {
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
        std::vector<int> devices = std::vector<int>{};
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
    Logger::debug("~Dispatcher:: Deconstructor calling deviceInfo->freeDevice()");
	deviceInfo->freeDevice();
}

// Updates the Places stored on CPU
std::vector<Place**> Dispatcher::refreshPlaces(int handle) {
    Logger::debug("Entering Dispatcher::refreshPlaces");
    PlacesModel *placesModel = model->getPlacesModel(handle);
    if (initialized) {
        int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(handle);
        int* dims = deviceInfo->getDimSize();
        std::vector<int> devices = deviceInfo->getDevices();
        int idxBump = 0;
        Logger::debug("Dispatcher::refreshPlaces: Initialized -> copying info from GPU to CPU");
        std::vector<void*> devStates = deviceInfo->getPlaceStates(handle);
        int placesStrideDev = deviceInfo->getPlacesStride(handle);
        int stateSize = placesModel->getStateSize();
        Logger::debug("Dispatcher::refreshPlaces: devPtrs size = %d; placesStrideDev = %d; stateSize = %d, idxBump = %d", devStates.size(), placesStrideDev, stateSize, idxBump);
        for (int i = 0; i < devices.size(); ++i) {
            Logger::debug("Dispatcher::refreshPlaces: copy memory on device: %d", i);
            int bytes = stateSize * placesStrideDev;
            cudaSetDevice(devices.at(i));
            CATCH(cudaMemcpy(placesModel->getStatePtr(i), deviceInfo->getPlaceStatesForTransfer(handle, i), bytes, cudaMemcpyDefault));
            if (idxBump == 0) {
                idxBump = dims[0] * MAX_AGENT_TRAVEL;
            }
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
        int* dims = deviceInfo->getDimSize();
        int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(placeHandle);

        int idxBump = 0;
        for (int i = 0; i < devices.size(); ++i) {
            Logger::debug("Launching Dispatcher::callAllPlacesKernel() on device: %d", 
                    devices.at(i));
            cudaSetDevice(devices.at(i));
            Logger::debug("Dispatcher::callAllPlaces: device: %d; pdims[0]: %d, pdims[1]: %d", i, pDims[0].x, pDims[1].x);

            // load any necessary arguments
            void *argPtr = NULL;
            if (argument != NULL) {
                CATCH(cudaMalloc((void** ) &argPtr, argSize));
                CATCH(cudaMemcpy(argPtr, argument, argSize, H2D));
            }

            callAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(i), stride, functionId, idxBump, argPtr);
            CHECK();
            cudaDeviceSynchronize();
            if (argPtr != NULL) {
                Logger::debug("Dispatcher::callAllPlaces: Freeing device args.");
                cudaFree(argPtr);
            }

            if (idxBump == 0) {
                idxBump = dims[0] * MAX_AGENT_TRAVEL;
            }
        }

        deviceInfo->copyGhostPlaces(placeHandle, model->getPlacesModel(placeHandle)->getStateSize());
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
    std::vector<void*> devStates = deviceInfo->getPlaceStates(handle);
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(handle);
    Logger::debug("Kernel dims = gridDim { %d, %d, %d } and blockDim = { %d, %d, %d }", pDims[0].x, pDims[0].y, pDims[0].z, pDims[1].x, pDims[1].y, pDims[1].z);
    std::vector<int> devices = deviceInfo->getDevices();
    int stride = deviceInfo->getPlacesStride(handle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(handle);
    int* dims = deviceInfo->getDimSize();
    int placesStride = deviceInfo->getPlacesStride(handle);

    int idxStart;
    int idxEnd;
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        //Logger::debug("Dispatcher::ExchangeAllPlaces: nptrs = %d", stride + (dims[0] * 2));
        if (i == 0) { 
            idxStart = 0;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        } else if (i == devices.size() - 1) { // update params for last device
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = 0;
        } else  { // update params for middle ranks
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        }  
        exchangeAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(i), stride, idxStart, idxEnd, destinations->size());
        CHECK();
        cudaDeviceSynchronize();
    }

    deviceInfo->copyGhostPlaces(handle, model->getPlacesModel(handle)->getStateSize());
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
    int idxStart;
    int idxEnd;
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        // load any necessary arguments
        void *argPtr = NULL;
        if (argument != NULL) {
            CATCH(cudaMalloc((void** ) &argPtr, argSize));
		    CATCH(cudaMemcpy(argPtr, argument, argSize, H2D));
        }
        Logger::debug("DispatcherExchangeAllPlaces: nptrs = %d", stride);
        if (i == 0) { 
            idxStart = 0;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        } else if (i == devices.size() - 1) { // update params for last device
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = 0;
        } else  { // update params for middle ranks
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        } 
        exchangeAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(i), stride, idxStart, idxEnd, destinations->size(), functionId, argPtr);
        CHECK();
        cudaDeviceSynchronize();

        if (argPtr != NULL) {
            Logger::debug("Dispatcher::exchangeAllPlaces: Freeing device args.");
            cudaFree(argPtr);
        }
    }

    deviceInfo->copyGhostPlaces(handle, model->getPlacesModel(handle)->getStateSize());
    Logger::debug("Exiting Dispatcher::exchangeAllPlaces with functionId = %d as an argument", functionId);
} 

void Dispatcher::callAllAgents(int agentHandle, int functionId, void *argument, int argSize) {
    if (initialized) {
        Logger::debug("Dispatcher::callAllAgents: Calling all on agents[%d]. Function id = %d", agentHandle, functionId);
        std::vector<std::pair<dim3, dim3>> aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);
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
            callAllAgentsKernel<<<aDims.at(i).first, aDims.at(i).second>>>(agtsPtrs.at(i), strides[i], functionId, argPtr);
            CHECK();
            cudaDeviceSynchronize();
            if (argPtr != NULL) {
                Logger::debug("Dispatcher::callAllAgents: Freeing device args.");
                cudaFree(argPtr);
            }
        }

        Logger::debug("Exiting Dispatcher::callAllAgents()");
    }
}

void Dispatcher::terminateAgents(int agentHandle, int placeHandle) {

    std::vector<int> devices = deviceInfo->getDevices();
    std::vector<Agent**> agentDevPtrs = deviceInfo->getDevAgents(agentHandle);
    std::vector<void*> agentStatePtrs = deviceInfo->getAgentsState(agentHandle);
    std::vector<Place**> placeDevPtrs = deviceInfo->getDevPlaces(placeHandle);

    int* nAgentsDev = deviceInfo->getnAgentsDev(agentHandle);
    int* maxAgents = deviceInfo->getMaxAgents(agentHandle);
    int placesStride = deviceInfo->getPlacesStride(placeHandle);
    int* nDims = deviceInfo->getDimSize();

    int agentStateSize = model->getAgentsModel(agentHandle)->getStateSize();
    std::vector<std::pair<dim3, dim3>> aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(placeHandle);

    // compact Agent arrays
    for (int i = 0; i < devices.size(); ++i) {
        cudaSetDevice(devices.at(i));

        // 1. bool array to represent alive/dead Agent's
        bool* h_compact_flags[maxAgents[i]]; 
        bool* d_compact_flags = NULL;
        int flags_size = maxAgents[i] * sizeof(bool);
        CATCH(cudaMalloc((void**) &d_compact_flags, flags_size));
        CATCH(cudaMemcpy(d_compact_flags, h_compact_flags, flags_size, H2D));
        setFlagsKernel<<<aDims.at(i).first, aDims.at(i).second>>>(agentDevPtrs.at(i), maxAgents[i], d_compact_flags);
        
        // 2. count true (alive Agents) values in bool array
        void *dev_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        unsigned h_count_output[1];
        unsigned *dev_count_output = 0;
        CubDebugExit(cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_compact_flags, dev_count_output, maxAgents[i]));
        CubDebugExit(cudaMalloc(&dev_temp_storage, temp_storage_bytes));
        CubDebugExit(cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_compact_flags, dev_count_output, maxAgents[i]));
        CATCH(cudaMemcpy(h_count_output, dev_count_output, sizeof(int), D2H));
        CATCH(cudaFree(dev_temp_storage));

        // 3. count false (dead Agents) values in bool array up to count of total alive Agents
        unsigned* h_count_dead = NULL;
        *h_count_dead = 0;
        unsigned* dev_count_dead = NULL;
        CubDebugExit(cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_compact_flags, dev_count_dead, *dev_count_output));
        CATCH(cudaMemcpy(h_count_dead, dev_count_dead, sizeof(unsigned), D2H));

        // 4. new unsigned array of size false (dead Agents) count
        unsigned* h_dead_locations[*h_count_dead];

        // 5. write alive Agent locations from count total alive Agents up to maxAgents to new array
        unsigned* dev_dead_locations = NULL;
        unsigned location_idx[1];
        *location_idx = 0;
        CATCH(cudaMalloc((void**) &dev_dead_locations, sizeof(unsigned) * (*h_count_dead)));
        writeMovingAgentLocationsKernel<<<aDims.at(i).first, aDims.at(i).second>>>((AgentState*)agentStatePtrs.at(i), *h_count_output, maxAgents[i], dev_dead_locations, location_idx);

        // 6. swap memory objects of dead and alive Agents to compact alive Agents 
        unsigned* dev_index = NULL;
        unsigned h_index[1];
        *h_index = 0;
        CATCH(cudaMalloc(&dev_index, sizeof(unsigned)));
        CATCH(cudaMemcpy(dev_index, h_index, sizeof(unsigned), H2D));
        compactAgentsKernel<<<aDims.at(i).first, aDims.at(i).second>>>((AgentState*)agentStatePtrs.at(i), *h_count_output, agentStateSize, dev_dead_locations, dev_index);

        // update each Place's agent array
        int ghostPlace_offset = 0;
        if (i != 0) {
            ghostPlace_offset = nDims[0];
        } 

        // 7. removes all Agents from Places because the pointers now point at the incorrect Agent
        unattachPlaceAgentsKernel<<<pDims[0], pDims[1]>>>(placeDevPtrs.at(i), placesStride, ghostPlace_offset);
        
        // 8. adds Agents back to correct Place based on Place index stored in AgentState
        reattachPlaceAgentsKernel<<<pDims[0], pDims[1]>>>(agentDevPtrs.at(i), placeDevPtrs.at(i), nAgentsDev[i]);
    }       
}

void Dispatcher::migrateAgents(int agentHandle, int placeHandle) {
    Logger::debug("Inside Dispatcher:: migrateAgents().");
    std::vector<Place**> p_ptrs = deviceInfo->getDevPlaces(placeHandle);
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(placeHandle);

    std::vector<Agent**> a_ptrs = deviceInfo->getDevAgents(agentHandle);
    std::vector<void*> a_ste_ptrs = deviceInfo->getAgentsState(agentHandle);
	std::vector<std::pair<dim3, dim3>> aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);

	std::vector<std::pair<Place**, void*>> gh_ptrs = deviceInfo->getTopGhostPlaces(placeHandle);
    std::vector<int> devices = deviceInfo->getDevices();
    int placeStride = deviceInfo->getPlacesStride(placeHandle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(placeHandle);
    int ghostPlaces = deviceInfo->getDimSize()[0] * MAX_AGENT_TRAVEL;
    int* nAgentsDev = deviceInfo->getnAgentsDev(agentHandle);

    Logger::debug("Dispatcher::MigrateAgents: number of places: %d", deviceInfo->getPlaceCount(placeHandle));
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Dispatcher::migrateAgents: Starting Long Distance Agent migration on device %d", i);
        for (int j = 0; j < devices.size(); ++j) {
            cudaSetDevice(devices.at(i)); 
            Logger::debug("Dispatcher::migrateAgents: longDistanceMigrationKernel copying to device %d", j);
            longDistanceMigrationKernel<<<aDims.at(i).first, aDims.at(i).second>>>(a_ptrs.at(i), 
                    a_ptrs.at(j), (AgentState*)a_ste_ptrs.at(i), (AgentState*)a_ste_ptrs.at(j), nAgentsDev[i], &nAgentsDev[j],
                    j, placeStride, model->getAgentsModel(agentHandle)->getStateSize());
        }
    }

    Logger::debug("Dispatcher::migrateAgents: Adding long distance migrated Agents to target Place.");
    for (int i = 0; i < devices.size(); ++i) {
        cudaSetDevice(devices.at(i));
        longDistanceMigrationsSetPlaceKernel<<<aDims.at(i).first, aDims.at(i).second>>>(p_ptrs.at(i), a_ptrs.at(i), 
                nAgentsDev[i], placeStride, ghostPlaces, ghostPlaceMult[i], devices.at(i));
    }
    
    Logger::debug("Dispatcher::resolveMigrationConflictsKernel() dims = gridDim %d and blockDim = %d", pDims[0].x, pDims[1].x);

    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher:: resolveMigrationConflictsKernel() on device: %d", devices.at(i));
        cudaSetDevice(devices.at(i));
        resolveMigrationConflictsKernel<<<pDims[0], pDims[1]>>>((gh_ptrs.at(i)).first, placeStride);
        CHECK();
        cudaDeviceSynchronize();		
    }

    Logger::debug("Dispatcher::migrateAgents: Number of place arrays in devPlaceMap == %d", deviceInfo->devPlacesMap.size());
    Logger::debug("Dispatcher::MigrateAgents: number of agents: %d", getNumAgents(agentHandle));
    for (int i = 0; i < devices.size(); ++i) {
        Logger::debug("Launching Dispatcher:: updateAgentLocationsKernel() on device: %d with number of agents = %d", devices.at(i), nAgentsDev[i]);
        cudaSetDevice(devices.at(i));
        updateAgentLocationsKernel<<<aDims.at(i).first, aDims.at(i).second>>>(a_ptrs.at(i), nAgentsDev[i]);
        CHECK();
        cudaDeviceSynchronize();
    }

	// TODO: Wait on even devices to finish moving Agent's locally
    //check each devices Agents for agents needing to move devices
    // TODO: Refactor to check if Agents needing to move devices have spawning to do.
    //       a. Do we leave them after local migration and spawn? 
    //       b. Do we leave them at origination for spawn
    //       c. Do we spawn and then migrate? ** THIS ONE **
    for (int i = 0; i < devices.size(); ++i) {
		cudaSetDevice(devices.at(i));
        if (i % 2 == 0) {
            // check right ghost stripe for Agents needing to move
            moveAgentsDownKernel<<<aDims.at(i).first, aDims.at(i).second>>>
                    (a_ptrs.at(i), a_ptrs.at(i + 1), (AgentState*)(a_ste_ptrs.at(i)), 
					(AgentState*)(a_ste_ptrs.at(i + 1)),
                    p_ptrs.at(i), p_ptrs.at(i + 1), i, placeStride, 
                    ghostPlaces, ghostPlaceMult[i], nAgentsDev[i], &(nAgentsDev[i + 1]), 
                    model->getAgentsModel(agentHandle)->getStateSize());
			CHECK();
            if (i != 0) {
                // check left ghost stripe for Agents needing to move
                moveAgentsUpKernel<<<aDims.at(i).first, aDims.at(i).second>>>
                        (a_ptrs.at(i), a_ptrs.at(i - 1), (AgentState*)(a_ste_ptrs.at(i)), 
						((AgentState*)a_ste_ptrs.at(i - 1)),
                        p_ptrs.at(i), p_ptrs.at(i - 1), i, placeStride, 
                        ghostPlaces, ghostPlaceMult[i], nAgentsDev[i], &(nAgentsDev[i - 1]), 
                        model->getAgentsModel(agentHandle)->getStateSize());
				CHECK();
            }
			
			cudaDeviceSynchronize();
        }

        else {
			// TODO: Wait on EVEN devices to finish moving agents globally
			if (i != devices.size() - 1) {
                // check right ghost stripe for Agents needing to move
                moveAgentsDownKernel<<<aDims.at(i).first, aDims.at(i).second>>>
                        (a_ptrs.at(i), a_ptrs.at(i + 1), (AgentState*)(a_ste_ptrs.at(i)), 
						(AgentState*)(a_ste_ptrs.at(i + 1)),
                        p_ptrs.at(i), p_ptrs.at(i + 1), i, placeStride, 
                        ghostPlaces, ghostPlaceMult[i], nAgentsDev[i], &(nAgentsDev[i + 1]),
                        model->getAgentsModel(agentHandle)->getStateSize());
				CHECK();
            }

            // check left ghost stripe for Agents needing to move
            moveAgentsUpKernel<<<aDims.at(i).first, aDims.at(i).second>>>
                    (a_ptrs.at(i), a_ptrs.at(i - 1), (AgentState*)(a_ste_ptrs.at(i)), 
					(AgentState*)(a_ste_ptrs.at(i - 1)),
                    p_ptrs.at(i), p_ptrs.at(i - 1), i, placeStride, 
                    ghostPlaces, ghostPlaceMult[i], nAgentsDev[i], &(nAgentsDev[i - 1]), 
                    model->getAgentsModel(agentHandle)->getStateSize());
			CHECK();
			cudaDeviceSynchronize();
        }
    }

	// TODO: Wait on ODD devices 
	// update total number of live agents
	int sumAgents = 0;
	for (int i = 0; i < devices.size(); ++i) {
		sumAgents += nAgentsDev[i];
	}
    deviceInfo->devAgentsMap[agentHandle].nAgents = sumAgents;

	// Check ghostPlaces for traveled Agents and update pointers
	for (int i = 1; i < devices.size(); ++i) {
		cudaSetDevice(devices.at(i));
		updateAgentPointersMovingDown<<<aDims.at(i).first, aDims.at(i).second>>>(p_ptrs.at(i), a_ptrs.at(i), 
				nAgentsDev[i], placeStride, ghostPlaces, ghostPlaceMult[i - 1], i);
		CHECK();
		cudaDeviceSynchronize();
	}

	for (int i = 0; i < devices.size() - 1; ++i) {
		cudaSetDevice(devices.at(i));
		updateAgentPointersMovingUp<<<aDims.at(i).first, aDims.at(i).second>>>(p_ptrs.at(i), a_ptrs.at(i),
				nAgentsDev[i], placeStride, ghostPlaces, ghostPlaceMult[i], i);
		CHECK();
	}

    Logger::debug("Exiting Dispatcher:: migrateAgents().");
}


void Dispatcher::spawnAgents(int handle) {
    
    Logger::debug("Inside Dispatcher::spawnAgents()");
    std::vector<Agent**> a_ptrs = deviceInfo->getDevAgents(handle);
    // TODO: get collected Agents
    std::vector<std::pair<dim3, dim3>> aDims = deviceInfo->getAgentsThreadBlockDims(handle);

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
        spawnAgentsKernel<<<aDims.at(i).first, aDims.at(i).second>>>(a_ptrs.at(i), numAgentObjects[i], deviceInfo->getMaxAgents(handle, i));
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

int Dispatcher::getAgentStateSize(int handle) {
    return model->getPlacesModel(handle)->getStateSize();
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

unsigned int* Dispatcher::calculateRandomNumbers(int size) {
    return deviceInfo->calculateRandomNumbers(size);
}

}// namespace mass

