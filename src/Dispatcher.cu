/**
 *  @file Dispatcher.cu
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#define COMPUTE_CAPABILITY_MAJOR 3

#include "Agents.h"
#include "Command.h"
#include "Dispatcher.h"
#include "Model.h"
#include "Places.h"
#include "Slice.h"

namespace mass {

Dispatcher::Dispatcher() {
	// do nothing
}

void Dispatcher::init(int ngpu, Model *model) {

	if (0 == ngpu) { // use all available GPU resources
		// adapted from the Cuda Toolkit Documentation: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
		cudaGetDeviceCount(&ngpu);
		for (int device = 0; device < ngpu; ++device) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device);
			printf("Device %d has compute capability %d.%d.\n", device,
					deviceProp.major, deviceProp.minor);
			if (COMPUTE_CAPABILITY_MAJOR == deviceProp.major) {
				// use this GPU
			}
		}
	} else { // use only specified GPU qty

	}
}

Dispatcher::~Dispatcher() {
	// destroy streams
	// destroy events
	// delete devices array
}

void Dispatcher::refreshPlaces(int handle) {
	// TODO get the unload the slices for this handle from the GPU without deleting
}

void Dispatcher::callAllPlaces(int handle, int functionId, void *argument,
		int argSize) {
	// TODO issue call
}

void *Dispatcher::callAllPlaces(int handle, int functionId, void *arguments[],
		int argSize, int retSize) {
	// TODO issue call
}

void Dispatcher::exchangeAllPlaces(int handle, int functionId,
		std::vector<int*> *destinations) {
	// TODO issue call
}

void Dispatcher::exchangeBoundaryPlaces(int handle) {
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

//	int ngpu;                   // number of GPUs in use
//	int* devices;               // array of GPU device ids
//	cudaStream_t* streams;      // cuda execution streams, two per device
//	cudaEvent_t* events; // cuda events to synchronize execution streams, one per device
//	Model *model; // the data model for this simulation

}// namespace mass

