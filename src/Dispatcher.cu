
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
__constant__ int nNeighbors_device; 


using namespace std;

namespace mass {

__global__ void callAllPlacesKernel(Place **ptrs, int nptrs, int functionId,
		void *argPtr) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < nptrs) {
		ptrs[idx]->callMethod(functionId, argPtr);
	}
}

/**
 * neighbors is converted into a 1D offset of relative indexes before calling this function
 */
__global__ void setNeighborPlacesKernel(Place **ptrs, int nptrs) {
	int idx = getGlobalIdx_1D_1D();

    if (idx < nptrs) {
        PlaceState *state = ptrs[idx]->getState();
        int nSkipped = 0;
        for (int i = 0; i < nNeighbors_device; ++i) {
            int j = idx + offsets_device[i];
            if (j >= 0 && j < nptrs) {
                state->neighbors[i - nSkipped] = ptrs[j];
                state->inMessages[i - nSkipped] = ptrs[j]->getMessage();
            } else {
                nSkipped++;
            }
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
		void *devPtr = deviceInfo->getPlaceState(handle);

        int stateSize = placesModel->getStateSize();
		int qty = placesModel->getNumElements();
		int bytes = stateSize * qty;
		CATCH(cudaMemcpy(((Place*) placesModel->getPlaceElements())->getState(), devPtr, bytes, D2H));


		Logger::debug("Exiting Dispatcher::refreshPlaces");
	}

	return placesModel->getPlaceElements();
}

void Dispatcher::callAllPlaces(int placeHandle, int functionId, void *argument, int argSize) {
	if (initialized) {
		Logger::debug("Dispatcher::callAllPlaces: Calling all on places[%d]. Function id = %d", placeHandle, functionId);

		// if (partInfo == NULL) { // the partition needs to be loaded
		// 	deviceInfo->loadPlacesModel(model->getPlacesModel(placeHandle));
		// } 

		// load any necessary arguments
		void *argPtr = NULL;
		if (argument != NULL) {
			deviceInfo->load(argPtr, argument, argSize);
			Logger::debug("Dispatcher::callAllPlaces: Loaded device\n");
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
	Logger::debug("Inside Dispatcher::updateNeighborhood");
	if (vec == neighborhood) { //no need to update
		Logger::print("Dispatcher::updateNeighborhood: Skipped the update, as neighborhood is up to date\n");
        return false;
    }

    Logger::print("Dispatcher::updateNeighborhood: Updating the neighborhood as it is new/changed\n");
    neighborhood = vec;
    int nNeighbors = vec->size();

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
    }
    
    // Now copy offsets to the GPU:
    cudaMemcpyToSymbol(offsets_device, offsets, sizeof(int) * nNeighbors);
    CHECK();
    cudaMemcpyToSymbol(nNeighbors_device, &nNeighbors, sizeof(int));
    CHECK();

    delete [] offsets;
    Logger::debug("Exiting Dispatcher::updateNeighborhood");
    return true;
}

void Dispatcher::exchangeAllPlaces(int handle, std::vector<int*> *destinations) {
	Logger::debug("Inside Dispatcher::exchangeAllPlaces");
	updateNeighborhood(handle, destinations);

	Place** ptrs = deviceInfo->getDevPlaces(handle);
	int nptrs = deviceInfo->countDevPlaces(handle);
	PlacesModel *p = model->getPlacesModel(handle); //only for getting blockDim and threadDim

	Logger::debug("Launching Dispatcher::setNeighborPlacesKernel()");
	setNeighborPlacesKernel<<<p->blockDim(), p->threadDim()>>>(ptrs, nptrs);
	CHECK();
	Logger::debug("Exiting Dispatcher::exchangeAllPlaces");
}

void Dispatcher::unloadDevice(DeviceConfig *device) {
	Logger::debug("Inside Dispatcher::unloadDevice\n");
    std::map<int, PlacesModel*> placesModels = model -> getAllPlacesModels();
	if (!placesModels.empty()) {
		map<int, PlacesModel*>::iterator itP = placesModels.begin();
		while (itP != placesModels.end()) {
			refreshPlaces(itP->first); //copy all stuff from GPU to CPU to PlaceState*
		}

		deviceInfo = NULL;
	}
}

}// namespace mass

