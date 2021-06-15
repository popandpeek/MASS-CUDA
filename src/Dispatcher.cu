
#include <sstream>
#include <algorithm>  // array compare
#include <iterator>
#include <typeinfo>
#include "../cub-1.8.0/cub/cub.cuh"
#include <omp.h>

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

        ptrs[idx + idxStart]->callMethod(functionId, argPtr);
    }
}

__global__ void setFlagsKernel(Agent** a_ptrs, int nPtrs, int* flags) {
    int idx = getGlobalIdx_1D_1D();
    if (idx < nPtrs) {
        if (a_ptrs[idx]->isAlive()) {
            flags[idx] = 1;
            return;
        }
        flags[idx] = 0;

    }
}

__global__ void writeAliveAgentLocationsKernel(Agent** a_ptrs, int startIdx, int qty, int* locations, int* locations_loc) {
    int idx = getGlobalIdx_1D_1D() + startIdx;
    if (idx < qty) {
        if(a_ptrs[idx]->isAlive()) {
            int location_idx = atomicAdd(locations_loc, 1);
            locations[location_idx] = idx;
        }
    }
}

__global__ void compactAgentsKernel(Agent** a_ptrs, int qty, int agentStateSize, int* locations, int* locations_loc) {
    int idx = getGlobalIdx_1D_1D();
    if (idx < qty) {
        if (!(a_ptrs[idx]->isAlive())) {
            // Copy alive Agent to dead Agent location
            int location_idx = atomicAdd(locations_loc, 1);
            memcpy(a_ptrs[idx]->state, a_ptrs[locations[location_idx]]->state, agentStateSize);
            a_ptrs[idx]->setIndex(idx);
            
            // remove Agent at old location from Place
            Place* pl = a_ptrs[locations[location_idx]]->getPlace();
            pl->removeAgent(a_ptrs[locations[location_idx]]);

            // add Agent at new location to Place
            pl->addAgent(a_ptrs[idx]);
        }
    }
}

__global__ void longDistanceMigrationKernel(Agent **src_agent_ptrs, Agent **dest_agent_ptrs, 
        AgentState *src_agent_state, AgentState *dest_agent_state, int nAgentsDevSrc, 
        int* nAgentsDevDest, int destDevice, int placesStride, int stateSize) {
    
    int idx = getGlobalIdx_1D_1D();
    if (idx < nAgentsDevSrc) {
        if (src_agent_ptrs[idx]->isAlive() && src_agent_ptrs[idx]->longDistanceMigration()) {
            int destPlaceIdx = src_agent_state[idx].destPlaceIdx;
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
        if (place_index >= (placesStride + (placesStride * device) + 
                (ghostPlaceMult * ghostPlaces - ghostPlaces))) {
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
			int placePtrIdx = agentPtrs[idx]->getPlaceDevIndex() - (device * placesStride) + (ghostPlaces + ghostPlaces * ghostSpaceMult);
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
			int placePtrIdx = agentPtrs[idx]->getPlaceDevIndex() - (device * placesStride) + ((ghostPlaces * 2) - (ghostSpaceMult * ghostPlaces));
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
                ptrs[idxStart+i]->setAlive(true);
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
        int gpuCount = 0;
        cudaGetDeviceCount(&gpuCount);
        
		if (gpuCount < 1) {
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

        omp_set_dynamic(0);
        omp_set_num_threads(devices.size());
        #pragma omp parallel 
        {
		    int gpu_id = -1;
            const int thread_id = omp_get_thread_num();
            CATCH(cudaSetDevice(thread_id));
		    CATCH(cudaGetDevice(&gpu_id));
		    Logger::debug("Thread id = %d selected device id = %d total threads = %d", thread_id, gpu_id, omp_get_num_threads());
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
        Logger::debug("Dispatcher::refreshPlaces: Initialized -> copying info from GPU to CPU");
        int placesStride = deviceInfo->getPlacesStride(handle);
        int stateSize = placesModel->getStateSize();
        
        #pragma omp parallel 
        {
            int gpu_id = -1;
            CATCH(cudaGetDevice(&gpu_id));
            Logger::debug("Dispatcher::refreshPlaces: copy memory on device: %d", gpu_id);
            int bytes = stateSize * placesStride;
            CATCH((cudaMemcpy(placesModel->getStatePtr(gpu_id), deviceInfo->getPlaceStatesForTransfer(handle, gpu_id), bytes, cudaMemcpyDefault)));
        }
	}
    
    Logger::debug("Exiting Dispatcher::refreshPlaces");

    return placesModel->getPlaceElements();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument, int argSize) {
	if (initialized) {
        Logger::debug("Dispatcher::callAllPlaces: Calling all on places[%d]. Function id = %d", 
                placeHandle, functionId);

        std::vector<Place**> devPtrs = deviceInfo->getDevPlaces(placeHandle); 
        int placesStride = deviceInfo->getPlacesStride(placeHandle);
        dim3* pDims = deviceInfo->getPlacesThreadBlockDims(placeHandle);

        #pragma omp parallel 
        {
            int gpu_id = -1;
            CATCH(cudaGetDevice(&gpu_id));
            Logger::debug("Dispatcher::callAllPlaces: device: %d; pdims[0]: %d, pdims[1]: %d", gpu_id, pDims[0].x, pDims[1].x);

            // load any necessary arguments
            void *argPtr = NULL;
            if (argument != NULL) {
                int devArgSize = calculatePlaceArgumentArrayChunkSize(placeHandle, argSize, gpu_id);
                Logger::debug("Dispatcher::callAllPlaces: Argument size = %d; devArgSize = %d", argSize, devArgSize);
                CATCH(cudaMalloc((void** ) &argPtr, devArgSize));
                CATCH(cudaMemcpy(argPtr, calculateArgumentPointer(placeHandle, gpu_id, argument, argSize), devArgSize, H2D));
            }
            
            int idxBump = gpu_id > 0 ? deviceInfo->getDimSize()[0] * MAX_AGENT_TRAVEL : 0;
            callAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(gpu_id), placesStride, functionId, idxBump, argPtr);
            CHECK();
            cudaDeviceSynchronize();
        
            if (argPtr != NULL) {
                Logger::debug("Dispatcher::callAllPlaces: Freeing device args.");
                cudaFree(argPtr);
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
    #pragma omp parallel shared(offsets_device, offsets)
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        cudaMemcpyToSymbol(offsets_device, offsets, sizeof(int) * nNeighbors);
        CHECK();
        Logger::debug("Copied constant memory to device %d", gpu_id);
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
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(handle);
    Logger::debug("Kernel dims = gridDim { %d, %d, %d } and blockDim = { %d, %d, %d }", pDims[0].x, pDims[0].y, pDims[0].z, pDims[1].x, pDims[1].y, pDims[1].z);
    int* dims = deviceInfo->getDimSize();
    int placesStride = deviceInfo->getPlacesStride(handle);

    #pragma omp parallel 
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", gpu_id);
        int idxStart;
        int idxEnd;
        if (gpu_id == 0) { 
            idxStart = 0;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        } else if (gpu_id == omp_get_num_threads() - 1) { // update params for last device
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = 0;
        } else  { // update params for middle ranks
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        }  
        exchangeAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(gpu_id), placesStride, idxStart, idxEnd, destinations->size());
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
    int placesStride = deviceInfo->getPlacesStride(handle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(handle);
    int* dims = deviceInfo->getDimSize();
    Logger::debug("Kernel dims = gridDim { %d, %d, %d } and blockDim = { %d, %d, %d }", pDims[0].x, pDims[0].y, pDims[0].z, pDims[1].x, pDims[1].y, pDims[1].z);
    
    #pragma omp parallel 
    {
        int idxStart;
        int idxEnd;
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        Logger::debug("Launching Dispatcher::exchangeAllPlacesKernel() on device: %d", gpu_id);

        // load any necessary arguments
        void *argPtr = NULL;
        if (argument != NULL) {
            int devArgSize = calculatePlaceArgumentArrayChunkSize(handle, argSize, gpu_id);
            Logger::debug("Dispatcher::callAllPlaces: Argument size = %d; devArgSize = %d", argSize, devArgSize);
            CATCH(cudaMalloc((void** ) &argPtr, devArgSize));
            CATCH(cudaMemcpy(argPtr, calculateArgumentPointer(handle, gpu_id, argument, argSize), devArgSize, H2D));
        }

        Logger::debug("DispatcherExchangeAllPlaces: nptrs = %d", placesStride);
        if (gpu_id == 0) { 
            idxStart = 0;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        } else if (gpu_id == omp_get_num_threads() - 1) { // update params for last device
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = 0;
        } else  { // update params for middle ranks
            idxStart = dims[0] * MAX_AGENT_TRAVEL;
            idxEnd = dims[0] * MAX_AGENT_TRAVEL;
        } 
        exchangeAllPlacesKernel<<<pDims[0], pDims[1]>>>(devPtrs.at(gpu_id), placesStride, idxStart, idxEnd, destinations->size(), functionId, argPtr);
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

        #pragma omp parallel
        {
            int gpu_id = -1;
            CATCH(cudaGetDevice(&gpu_id));
            Logger::debug("Launching Dispatcher::callAllAgentsKernel() on device: %d", gpu_id);

            // load any necessary arguments
            void *argPtr = NULL;
            if (argument != NULL) {
                int devArgSize = argSize / omp_get_num_threads();
                char* tmp = (char*)argument;
                CATCH(cudaMalloc((void** ) &argPtr, devArgSize));
                CATCH(cudaMemcpy(argPtr, (void*)(&(tmp[gpu_id * devArgSize])), devArgSize, H2D));
            }

            callAllAgentsKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(agtsPtrs.at(gpu_id), strides[gpu_id], functionId, argPtr);
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
    Logger::debug("Launching Dispatcher::terminateAgents()");
    std::vector<int> devices = deviceInfo->getDevices();
    std::vector<Agent**> agentDevPtrs = deviceInfo->getDevAgents(agentHandle);
    std::vector<void*> agentStatePtrs = deviceInfo->getAgentsState(agentHandle);
    std::vector<Place**> placeDevPtrs = deviceInfo->getDevPlaces(placeHandle);

    int* nAgentsDev = deviceInfo->getnAgentsDev(agentHandle);
    int maxAgents = deviceInfo->getMaxAgents(agentHandle);
    int placesStride = deviceInfo->getPlacesStride(placeHandle);
    int* nDims = deviceInfo->getDimSize();

    int agentStateSize = model->getAgentsModel(agentHandle)->getStateSize();
    std::vector<std::pair<dim3, dim3>> aDims = deviceInfo->getAgentsThreadBlockDims(agentHandle);
    dim3* pDims = deviceInfo->getPlacesThreadBlockDims(placeHandle);

    // compact Agent arrays
    #pragma omp parallel
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        Logger::debug("########################  DEVICE #%d ########################", gpu_id);
        int agent_mem_used = (int)(((float)nAgentsDev[gpu_id] / (float)maxAgents) * 100);
        if (agent_mem_used > AGENT_MEM_CHECK) {
            // 1. bool array to represent alive/dead Agent's
            int h_compact_flags[nAgentsDev[gpu_id]]; 
            int* d_compact_flags;
            CATCH(cudaMalloc((void**) &d_compact_flags, (nAgentsDev[gpu_id] * sizeof(int))));
            Logger::debug("Dispatcher::terminateAgents(): alive Agents = %d, maxAgents = %d.", nAgentsDev[gpu_id], maxAgents);
            setFlagsKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(agentDevPtrs.at(gpu_id), nAgentsDev[gpu_id], d_compact_flags);
            Logger::debug("Dispatcher::terminateAgents(): Set Flags kernel completes.");
            CATCH(cudaMemcpy(h_compact_flags, d_compact_flags, nAgentsDev[gpu_id] * sizeof(int), D2H));
            // for (int j = 0; j < nAgentsDev[i]; ++j) {
            //     Logger::debug("Index = %d : Flag = %d", j, h_compact_flags[j]);
            // }

            // 2. count true (alive Agents) values in bool array
            void *dev_temp_storage;
            size_t temp_storage_bytes = 0;
            int h_count_alive_total[1];
            int *d_count_alive_total;
            CATCH(cudaMalloc((void**) &d_count_alive_total, sizeof(int)));
            CubDebug(cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_compact_flags, d_count_alive_total, nAgentsDev[gpu_id]));
            CubDebug(cudaMalloc(&dev_temp_storage, temp_storage_bytes));
            CubDebug(cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_compact_flags, d_count_alive_total, nAgentsDev[gpu_id]));
            CATCH(cudaMemcpy(h_count_alive_total, d_count_alive_total, sizeof(int), D2H));
            CATCH(cudaFree(d_count_alive_total));
            CATCH(cudaFree(dev_temp_storage));
            // Logger::debug("Dispatcher::terminateAgents(): CUB::DeviceReduce::Sum for total alive Agents completes = %d", *h_count_alive_total);
            
            if (*h_count_alive_total < nAgentsDev[gpu_id]) {
                // 3. count alive Agents values in flag array up to count of total alive Agents
                int h_count_alive_left[1];
                *h_count_alive_left = 0;
                int* d_count_alive_left;
                temp_storage_bytes = 0;
                void *dev_temp_storage2;
                CATCH(cudaMalloc((void**) &d_count_alive_left, sizeof(int)));
                CubDebug(cub::DeviceReduce::Sum(dev_temp_storage2, temp_storage_bytes, d_compact_flags, d_count_alive_left, *h_count_alive_total));
                CubDebug(cudaMalloc(&dev_temp_storage2, temp_storage_bytes));
                CubDebug(cub::DeviceReduce::Sum(dev_temp_storage2, temp_storage_bytes, d_compact_flags, d_count_alive_left, *h_count_alive_total));
                CATCH(cudaMemcpy(h_count_alive_left, d_count_alive_left, sizeof(int), D2H));
                CATCH(cudaFree(d_count_alive_left));
                CATCH(cudaFree(dev_temp_storage2));

                // Logger::debug("Dispatcher::terminateAgents(): CUB::DeviceReduce::Sum for alive Agents left of pivot completes = %d", *h_count_alive_left);

                // 4. new int array of size false (dead Agents) count
                int arrSize = *h_count_alive_total - *h_count_alive_left;
                Logger::debug("Value of arrSize = %d", arrSize);
                if (arrSize != 0) {
                    // 5. write alive Agent locations from count total alive Agents up to nAgentsDev to new array
                    int* dev_loc_index;
                    int* d_live_locations;
                    int location_idx[1];
                    *location_idx = 0;
                    CATCH(cudaMalloc((void**) &dev_loc_index, sizeof(int)));
                    CATCH(cudaMemcpy(dev_loc_index, location_idx, sizeof(int), H2D));
                    CATCH(cudaMalloc((void**) &d_live_locations, sizeof(int) * arrSize));
                    writeAliveAgentLocationsKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(agentDevPtrs.at(gpu_id), *h_count_alive_total, nAgentsDev[gpu_id], d_live_locations, dev_loc_index);
                    CHECK();
                    Logger::debug("Dispatcher::terminateAgents(): writeEmptyAgentLocationsKernel completes.");
                    int h_liveLocations[arrSize];
                    CATCH(cudaMemcpy(location_idx, dev_loc_index, sizeof(int), D2H));
                    CATCH(cudaMemcpy(h_liveLocations, d_live_locations, sizeof(int) * arrSize, D2H));
                    // Logger::debug("** arrSize = %d : dev_loc_idx = %d **", arrSize, *location_idx);
                    // for (int j = 0; j < arrSize; ++j) {
                    //     Logger::debug("***** Index = %d : Flag = %d", j, h_liveLocations[j]);
                    // }

                    // 6. swap memory objects of dead and alive Agents to compact alive Agents 
                    int* dev_index;
                    int h_index[1];
                    *h_index = 0;
                    CATCH(cudaMalloc((void**) &dev_index, sizeof(int)));
                    CATCH(cudaMemcpy(dev_index, h_index, sizeof(int), H2D));
                    compactAgentsKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(agentDevPtrs.at(gpu_id), *h_count_alive_total, agentStateSize, d_live_locations, dev_index);
                    CATCH(cudaMemcpy(h_index, dev_index, sizeof(int), D2H));
                    // Logger::debug("*** Alive Agents moved = %d : Dead Agents copied over = %d", *h_index, *location_idx);
                    // Logger::debug("Dispatcher::terminateAgents(): compactAgentsKernel completes.");
                    CATCH(cudaFree(dev_loc_index));
                    CATCH(cudaFree(dev_index));
                    CATCH(cudaFree(d_live_locations));

                    deviceInfo->setnAgentsDev(agentHandle, gpu_id, *h_count_alive_total);
                    // Logger::debug("Dispatcher::terminateAgents(): setnAgentsDev() -> %d", *h_count_alive_total);
                }
            }
            CATCH(cudaFree(d_compact_flags));
        }
        cudaDeviceSynchronize();
    }       

    Logger::debug("Exiting Dispatcher::terminateAgents()");
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
    int placesStride = deviceInfo->getPlacesStride(placeHandle);
    int* ghostPlaceMult = deviceInfo->getGhostPlaceMultiples(placeHandle);
    int ghostPlaces = deviceInfo->getDimSize()[0] * MAX_AGENT_TRAVEL;
    int* nAgentsDev = deviceInfo->getnAgentsDev(agentHandle);

    // Logger::debug("Dispatcher::MigrateAgents: number of places: %d", deviceInfo->getPlaceCount(placeHandle));
    // for (int i = 0; i < devices.size(); ++i) {
    //     Logger::debug("Dispatcher::migrateAgents: Starting Long Distance Agent migration on device %d", i);
    //     for (int j = 0; j < devices.size(); ++j) {
    //         cudaSetDevice(devices.at(i)); 
    //         Logger::debug("Dispatcher::migrateAgents: longDistanceMigrationKernel copying to device %d", j);
    //         longDistanceMigrationKernel<<<aDims.at(i).first, aDims.at(i).second>>>(a_ptrs.at(i), 
    //                 a_ptrs.at(j), (AgentState*)a_ste_ptrs.at(i), (AgentState*)a_ste_ptrs.at(j), nAgentsDev[i], &nAgentsDev[j],
    //                 j, placeStride, model->getAgentsModel(agentHandle)->getStateSize());
    //     }
    // }

    // Logger::debug("Dispatcher::migrateAgents: Adding long distance migrated Agents to target Place.");
    // for (int i = 0; i < devices.size(); ++i) {
    //     cudaSetDevice(devices.at(i));
    //     longDistanceMigrationsSetPlaceKernel<<<aDims.at(i).first, aDims.at(i).second>>>(p_ptrs.at(i), a_ptrs.at(i), 
    //             nAgentsDev[i], placeStride, ghostPlaces, ghostPlaceMult[i], devices.at(i));
    // }
    
    Logger::debug("Dispatcher::resolveMigrationConflictsKernel() dims = gridDim %d and blockDim = %d", pDims[0].x, pDims[1].x);

    #pragma omp parallel
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        Logger::debug("Launching Dispatcher:: resolveMigrationConflictsKernel() on device: %d", gpu_id);
        resolveMigrationConflictsKernel<<<pDims[0], pDims[1]>>>((gh_ptrs.at(gpu_id)).first, placesStride);
        CHECK();		
    }

    Logger::debug("Dispatcher::migrateAgents: Number of place arrays in devPlaceMap == %d", deviceInfo->devPlacesMap.size());
    Logger::debug("Dispatcher::MigrateAgents: number of agents: %d", getNumAgents(agentHandle));
    #pragma omp parallel
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        Logger::debug("Launching Dispatcher:: updateAgentLocationsKernel() on device: %d with number of agents = %d", gpu_id, nAgentsDev[gpu_id]);
        updateAgentLocationsKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(a_ptrs.at(gpu_id), nAgentsDev[gpu_id]);
        CHECK();
    }

	// TODO: Wait on even devices to finish moving Agent's locally
    //check each devices Agents for agents needing to move devices
    // TODO: Refactor to check if Agents needing to move devices have spawning to do.
    //       a. Do we leave them after local migration and spawn? 
    //       b. Do we leave them at origination for spawn
    //       c. Do we spawn and then migrate? ** THIS ONE **
    #pragma omp parallel
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        if (gpu_id % 2 == 0) {
            // check bottom ghost stripe for Agents needing to move
            moveAgentsDownKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>
                    (a_ptrs.at(gpu_id), a_ptrs.at(gpu_id + 1), (AgentState*)(a_ste_ptrs.at(gpu_id)), 
					(AgentState*)(a_ste_ptrs.at(gpu_id + 1)),
                    p_ptrs.at(gpu_id), p_ptrs.at(gpu_id + 1), gpu_id, placesStride, 
                    ghostPlaces, ghostPlaceMult[gpu_id], nAgentsDev[gpu_id], &(nAgentsDev[gpu_id + 1]), 
                    model->getAgentsModel(agentHandle)->getStateSize());
			CHECK();
            if (gpu_id != 0) {
                // check top ghost stripe for Agents needing to move
                moveAgentsUpKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>
                        (a_ptrs.at(gpu_id), a_ptrs.at(gpu_id - 1), (AgentState*)(a_ste_ptrs.at(gpu_id)), 
						((AgentState*)a_ste_ptrs.at(gpu_id - 1)),
                        p_ptrs.at(gpu_id), p_ptrs.at(gpu_id - 1), gpu_id, placesStride, 
                        ghostPlaces, ghostPlaceMult[gpu_id], nAgentsDev[gpu_id], &(nAgentsDev[gpu_id - 1]), 
                        model->getAgentsModel(agentHandle)->getStateSize());
				CHECK();
            }
        }

        else {
			// TODO: Wait on EVEN devices to finish moving agents globally
			if (gpu_id != devices.size() - 1) {
                // check bottom ghost stripe for Agents needing to move
                moveAgentsDownKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>
                        (a_ptrs.at(gpu_id), a_ptrs.at(gpu_id + 1), (AgentState*)(a_ste_ptrs.at(gpu_id)), 
						(AgentState*)(a_ste_ptrs.at(gpu_id + 1)),
                        p_ptrs.at(gpu_id), p_ptrs.at(gpu_id + 1), gpu_id, placesStride, 
                        ghostPlaces, ghostPlaceMult[gpu_id], nAgentsDev[gpu_id], &(nAgentsDev[gpu_id + 1]),
                        model->getAgentsModel(agentHandle)->getStateSize());
				CHECK();
            }

            // check top ghost stripe for Agents needing to move
            moveAgentsUpKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>
                    (a_ptrs.at(gpu_id), a_ptrs.at(gpu_id - 1), (AgentState*)(a_ste_ptrs.at(gpu_id)), 
					(AgentState*)(a_ste_ptrs.at(gpu_id - 1)),
                    p_ptrs.at(gpu_id), p_ptrs.at(gpu_id - 1), gpu_id, placesStride, 
                    ghostPlaces, ghostPlaceMult[gpu_id], nAgentsDev[gpu_id], &(nAgentsDev[gpu_id - 1]), 
                    model->getAgentsModel(agentHandle)->getStateSize());
			CHECK();
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
	#pragma omp parallel
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
		updateAgentPointersMovingDown<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(p_ptrs.at(gpu_id), a_ptrs.at(gpu_id), 
				nAgentsDev[gpu_id], placesStride, ghostPlaces, ghostPlaceMult[gpu_id - 1], gpu_id);
		CHECK();
	}

	#pragma omp parallel 
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
		updateAgentPointersMovingUp<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(p_ptrs.at(gpu_id), a_ptrs.at(gpu_id),
				nAgentsDev[gpu_id], placesStride, ghostPlaces, ghostPlaceMult[gpu_id], gpu_id);
		CHECK();
	}

    Logger::debug("Exiting Dispatcher:: migrateAgents().");
}


void Dispatcher::spawnAgents(int handle) {
    
    Logger::debug("Inside Dispatcher::spawnAgents()");
    std::vector<Agent**> a_ptrs = deviceInfo->getDevAgents(handle);
    std::vector<std::pair<dim3, dim3>> aDims = deviceInfo->getAgentsThreadBlockDims(handle);

    std::vector<int> devices = deviceInfo->getDevices();
    int* nAgentsDevs = deviceInfo->getnAgentsDev(handle);
    int maxAgents = deviceInfo->getMaxAgents(handle);

    Logger::debug("Dispatcher::spawnAgents(): gets stuff from deviceInfo.");

    #pragma omp parallel 
    {
        int gpu_id = -1;
        CATCH(cudaGetDevice(&gpu_id));
        int h_numAgentObjects[1];
        *h_numAgentObjects = 0;
        int* d_numAgentObjects;
        CATCH(cudaMalloc((void**) &d_numAgentObjects, sizeof(int)));
        CATCH(cudaMemcpy(d_numAgentObjects, &nAgentsDevs[gpu_id], sizeof(int), H2D));
        Logger::debug("Dispatcher::spawnAgents(): Completes H2D memCpy's.");

        spawnAgentsKernel<<<aDims.at(gpu_id).first, aDims.at(gpu_id).second>>>(a_ptrs.at(gpu_id), d_numAgentObjects, maxAgents);
        CHECK();
        CATCH(cudaMemcpy(h_numAgentObjects, d_numAgentObjects, sizeof(int), D2H));
        CATCH(cudaFree(d_numAgentObjects));

        // Is this necessary? If so, may need to accumulate count of results above
        if (*(h_numAgentObjects) > maxAgents) {
            throw MassException("Trying to spawn more agents than the maximum set for the system");
        }

        Logger::debug("Dispatcher::spawnAgents(): Completes kernel calls.");
        int nNewAgents = *h_numAgentObjects - nAgentsDevs[gpu_id];
        Logger::debug("Dispatcher::spawnAgents(): Completes nNewAgents init and sub.");
        deviceInfo->devAgentsMap[handle].nAgents += nNewAgents;
        Logger::debug("Dispatcher::spawnAgents(): Completes nNewAgents set.");
        deviceInfo->devAgentsMap[handle].nAgentsDev[gpu_id] += nNewAgents;
        Logger::debug("Dispatcher::spawnAgents(): Completes nAgentsDev increase.");
    }

    Logger::debug("Finished Dispatcher::spawnAgents");
}

int Dispatcher::getNumAgents(int agentHandle) {
    return deviceInfo->getNumAgents(agentHandle);
}

int Dispatcher::getMaxAgents(int agentHandle) {
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

int Dispatcher::getNumAgentsInstantiated(int handle) {
    return deviceInfo->getMaxAgents(handle);
}

unsigned* Dispatcher::calculateRandomNumbers(int size) {
    return deviceInfo->calculateRandomNumbers(size);
}

unsigned Dispatcher::calculatePlaceArgumentArrayChunkSize(int placeHandle, int byteSize, int device) {
    unsigned argSizePerDevice = byteSize / deviceInfo->activeDevices.size();
    unsigned ghostPaddingArgSize = ((deviceInfo->getGhostPlaceMultiples(placeHandle)[0] * deviceInfo->getDimSize()[0] * MAX_AGENT_TRAVEL) * 
            (byteSize / (deviceInfo->activeDevices.size() * deviceInfo->getPlacesStride(placeHandle))));
    if (device == 0) {
        return argSizePerDevice;
    } else {
        return argSizePerDevice + ghostPaddingArgSize;
    }
}

void* Dispatcher::calculateArgumentPointer(int placeHandle, int device, void* arg, int argSize) {
    int argIdx = 0;
    int argsPerPlace = argSize / sizeof(int) / deviceInfo->getPlaceCount(placeHandle);
    Logger::debug("Dispatcher::calculateArgumentPointer: argsPerPlace: %d", argsPerPlace);
    if (device != 0) {
        argIdx = (device * deviceInfo->getPlacesStride(placeHandle)) - 
                (deviceInfo->getGhostPlaceMultiples(placeHandle)[0] * 
                deviceInfo->getDimSize()[0] * argsPerPlace * MAX_AGENT_TRAVEL);
    }

    char* tmp = (char*)arg;
    return ((void*)(&(tmp[argIdx * sizeof(int)])));
}
}// namespace mass

