/**
 *  @file AllTests.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <cuda_runtime.h>
#include <vector>

#include "../src/Logger.h"
#include "../src/Mass.h"
#include "../src/Places.h"
#include "../src/Place.h"
#include "../src/DllClass.h"
#include "../src/PlacesPartition.h"
#include "../src/DeviceConfig.h"
#include "../src/Dispatcher.h"
#include "../src/cudaUtil.h"

#include "AllTests.h"
#include "TestPlace.h"
#include "TestPlaceInstantiator.h"

using namespace mass;

__global__ void setPlacePtrsKernel(Place **ptrs, void *objs, int nPtrs,
		int nObjs, int Tsize) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < nObjs) {
		char* dest = ((char*) objs) + idx * Tsize;
		ptrs[idx] = (Place*) dest;
	} else if (idx < nPtrs) { // nObjs < idx < nPtrs
		ptrs[idx] = NULL;
	}
}

// TODO this kernel function isn't working. Don't know why.
__global__ void callAllPlacesKernel(TestPlace* ptrs, int nptrs, int functionId,
		void *argPtr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < nptrs) {
//		if(NULL != ptrs[idx]){
		ptrs[idx].callMethod(functionId, argPtr);
//		}
	}
}

template<typename T>
bool arrayequals(T *a, T *b, int size) {
	for (int i = 0; i < size; ++i) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

AllTests::AllTests() {
	failedTests = 0;
}

AllTests::~AllTests() {
}

bool AllTests::proofOfConcept() {

	int failures = 0;

	// manually use on the CPU
//	DllClass dll("libTestPlace.so");
//	TestPlace *proto = (TestPlace*) dll.instantiate(NULL);
//	Logger::debug("Proto place starting value = %d",
//			*((int*) proto->getMessage()));
//	proto->callMethod(TestPlace::SET_TO_ONE);
//	Logger::debug("Proto place SET_TO_ONE value = %d",
//			*((int*) proto->getMessage()));
//
//	Logger::debug("Creating an array.");
//	int size = 10;
//	Place *places[size];
//	TestPlace *objs = (TestPlace*) malloc(size * proto->placeSize());
//	char *dest = (char*) objs;
//	for (int i = 0; i < size; ++i) {
//		memcpy(dest, proto, proto->placeSize());
//		*((TestPlace*)dest) = *proto;
//		places[i] = (Place*) dest;
//		dest += proto->placeSize();
//	}
//
//	int arg = 42;
//	for (int i = 0; i < size; ++i) {
//		places[i]->callMethod(TestPlace::SET_TO_ARG, &arg);
//	}

	// rerun test on GPU
//	TestPlace arr[size];
//	TestPlace *dObjs;
//	CATCH(cudaMalloc((void**) &dObjs, sizeof(TestPlace) * size));
//	CATCH(cudaMemcpy(dObjs, arr, sizeof(TestPlace) * size, H2D));
//	dim3 block(1,1,1);
//	dim3 thread(size,1,1);
//	callAllPlacesKernel<<<block,thread>>>(dObjs, size, TestPlace::SET_TO_ONE, NULL);
//	CHECK();
//
//	CATCH(cudaMemcpy(arr, dObjs, sizeof(TestPlace) * size, cudaMemcpyDeviceToHost));
//
//
//	for(int i =  0; i < size; ++i){
//		if(*((int*) arr[i].getMessage()) != 1){
//			failures++;
//			Logger::warn("Call All did not work. ");
//		}
//	}
//	cudaFree(dObjs);
//	cudaFree(dPtrs);
//
//
//	// clean up
//	free(objs);
//	dll.destroy(proto);
//	failedTests += failures;
	return 0 == failures;
}

bool AllTests::runMassTests() {
	Logger::info("Beginning testing of Mass class.");
	int failures = 0;

	if (Mass::numAgentsInstances() != 0) {
		++failures;
		Logger::error(
				"Mass::numAgentsInstances() returning more than 0, but should be 0.");
	} else {
		Logger::info("Mass::numAgentsInstances() passed.");
	}

	if (Mass::numPlacesInstances() != 0) {
		++failures;
		Logger::error(
				"Mass::numPlacesInstances() returning more than 0, but should be 0.");
	} else {
		Logger::info("Mass::numPlacesInstances() passed.");
	}

	if (NULL != Mass::getAgents(0)) {
		++failures;
		Logger::error(
				"Mass::getAgents(0) is not returning NULL, but should.");
	} else {
		Logger::info("Mass::getAgents(0) passed.");
	}

	if (NULL != Mass::getPlaces(0)) {
		++failures;
		Logger::error(
				"Mass::getPlaces(0) is not returning NULL, but should.");
	} else {
		Logger::info("Mass::getPlaces(0) passed.");
	}

	Logger::info("Done testing of Mass class.");
	failedTests += failures;
	return 0 == failures;
}

bool AllTests::runDllClassTests() {
	int failures = 0;
//	int argLength = 1;
//	void arg[1] = {5};
//	Logger::info("Attempting to instantiate a place.");
//	Place *protoPlace = (Place *) (dllClass->instantiate(arg));
//	if (NULL != protoPlace) {
//		Logger::info("Place instantiation successful.");
//	} else {
//		++failures;
//		Logger::error(
//				"Place instantiation failed. Remaining dllClass tests aborted.");
//		return false;
//	}
//
//	int Tsize = protoPlace->placeSize();
//	if (sizeof(TestPlace) != Tsize) {
//		++failures;
//		Logger::error(
//				"TestPlace size function does not work. sizeof(TestPlace) = %d, Tsize = %d",
//				sizeof(TestPlace), Tsize);
//	} else {
//		Logger::info(
//				"TestPlace size function does works. sizeof(TestPlace) = %d, Tsize = %d",
//				sizeof(TestPlace), Tsize);
//	}
//
//	if (*((unsigned*) protoPlace->getMessage()) != arg[0]) {
//		++failures;
//		Logger::error("TestPlace getMessage() function does not work.");
//	} else {
//		Logger::info("TestPlace getMessage() function passed.");
//	}
//
//	//  space for an entire set of place instances
//	Logger::info("Allocating place elements.");
//	int numElements = 1;
//	for (int i = 0; i < argLength; ++i) {
//		numElements *= arg[i];
//	}
//	dllClass->placeElements = malloc(numElements * Tsize);
//	Logger::info("Done allocating place elements.");
//
//	// char is used to allow void* arithmatic in bytes
//	char *copyDest = (char*) dllClass->placeElements;
//	Place **places = new Place*[numElements];
//
//	Logger::info("Copying protoplace to each element.");
//	for (int i = 0; i < numElements; ++i) {
//		// memcpy protoplace
//		memcpy(copyDest, protoPlace, Tsize);
//		((Place *) copyDest)->callMethod(TestPlace::SET_TO_ARG, &i); // set the unique index
//		places[i] = (Place*) copyDest;
//		copyDest += Tsize; // increment copy destination
//	}
//
//	for (int i = 0; i < numElements; ++i) {
//		Logger::info("Evaluating places[%d]", i);
//		int msg = *((int*) places[i]->getMessage());
//		if (msg != i) {
//			++failures;
//			Logger::error("Place[%d] has wrong value.", i);
//		}
//	}
//
//	Logger::info("Copied all proto places successfully.");
//	Logger::info("Destroying the protoplace.");
//	dllClass->destroy(protoPlace); // we no longer need this
//	Logger::info("Proto place destroyed.");
//
//	delete dllClass;
//	delete[] places;
//	Logger::info("Done testing of DllClass.");
//	failedTests += failures;
	return 0 == failures;
}

bool AllTests::runPlacesTests() {
	Logger::info("Beginning testing of Places class.");
	int failures = 0;

	int arg = 5;
	int nDims = 1;
	int size[] = { 10 };

	TestPlaceInstantiator t;
	Places *p = Mass::createPlaces<TestPlaceInstantiator>(t, 0, &arg, sizeof(int), nDims, size, 0);

	if (p->getDimensions() != nDims) {
		++failures;
		Logger::error(
				"Places::getDimensions() returned incorrect number of dimensions.");
	} else {
		Logger::info("Places::getDimensions() passed.");
	}

	if (p->getHandle() != 0) {
		++failures;
		Logger::error("Places::getHandle() returned incorrect number.");
	} else {
		Logger::info("Places::getHandle() passed.");
	}

	int idx[1] = { 4 };
	int rmi = 4;
	if (p->getRowMajorIdx(idx) != rmi) {
		++failures;
		Logger::error(
				"Places::getRowMajorIdx() returned incorrect row major index.");
	} else {
		Logger::info("Places::getRowMajorIdx() passed.");
	}

	std::vector<int> vec = p->getIndexVector(rmi);
	if (!arrayequals(idx, &vec[0], vec.size())) {
		++failures;
		Logger::error(
				"Places::getIndexVector() returned incorrect index vector.");
	} else {
		Logger::info("Places::getIndexVector() passed.");
	}

	if (!arrayequals(p->size(), size, nDims)) {
		++failures;
		Logger::error(
				"Places::size() returned incorrect dimensions vector.");
	} else {
		Logger::info("Places::size() passed.");
	}

//	Place ** myPlaces; // = p->getElements();
//	if (NULL == myPlaces) {
//		++failures;
//		Logger::error("Places::getElements() returned NULL incorrectly.");
//	} else {
//		bool allOk = true;
//		for (int i = 0; i < p->getNumPlaces(); ++i) {
//			if (NULL == myPlaces[i]) {
//				++failures;
//				allOk = false;
//				Logger::error("Places::myPlaces[i] is NULL incorrectly.");
//			} else if (*((int*) myPlaces[i]->getMessage()) != arg) {
//				++failures;
//				allOk = false;
//				Logger::error(
//						"Place did not initialize with arg correctly.");
//			}
//		}
//		if(allOk){
//			Logger::info("Place constructor set argument correctly.");
//		}
//	}

	// run callAll test
	// TODO fix bug here! This isn't actually changing the value of the places.
	Logger::info("Executing SET_TO_ONE test.");
	p->callAll(TestPlace::SET_TO_ONE);
//	myPlaces = p->getElements();
//	for (int i = 0; i < p->getNumPlaces(); ++i) {
//		int val = *((int*) myPlaces[i]->getMessage());
//		if (val != 1) {
//			++failures;
//			Logger::error(
//					"Places.callAll(SET_TO_ONE) did not work. Value[%d] = %d",
//					i, val);
////			break;
//		}
//	}

//	Logger::info("Getting places manually...");
//
//	PlacesPartition *pp = p->getPartition(0);
//	DeviceConfig &d = Mass::dispatcher->loadedPlaces[pp];
//	d.setAsActiveDevice();
//
//	int nBytes = p->Tsize * size[0];
//	void* elems = malloc(nBytes);
//	CATCH(cudaMallocHost((void**) &elems, nBytes));
//
//	CATCH(cudaMemcpyAsync(elems,
//			pp->getLeftBuffer(),
//			nBytes,
//			cudaMemcpyDeviceToHost,
//			d.outputStream));
//
//	char *dst = (char*)  elems;
//	for(int i = 0; i < size[0]; ++i){
//		Place *tmp = (Place*) dst;
//		int val = *((int*) tmp->getMessage());
//		Logger::info("testplaces value[%d] = %d", i, val);
//		dst += p->Tsize;
//	}
//	cudaFreeHost(elems);
//
//	// do it manually and see if it works
//	TestPlace testplaces[10];
//	TestPlace *devPtr;
//	cudaMalloc((void**) devPtr, 10 * Tsize);
//	cudaMemcpy(devPtr, testPlaces, 10*Tsize, H2D);
//
//	delete p;

	Logger::info("Done testing of Places class.");
	failedTests += failures;
	return 0 == failures;
}

bool AllTests::runDispatcherTests() {

	return true;
}

bool AllTests::runPlacesPartitionTests() {
	return true;
}

bool AllTests::runAgentsTests() {
	return true;
}

bool AllTests::runAgentsPartitionTests() {
	return true;
}

