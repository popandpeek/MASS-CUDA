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
#include "../src/PlacesPartition.h"
#include "../src/DeviceConfig.h"
#include "../src/Dispatcher.h"
#include "../src/cudaUtil.h"

#include "AllTests.h"
#include "TestPlace.h"
#include "TestState.h"

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
		Logger::error("Mass::getAgents(0) is not returning NULL, but should.");
	} else {
		Logger::info("Mass::getAgents(0) passed.");
	}

	if (NULL != Mass::getPlaces(0)) {
		++failures;
		Logger::error("Mass::getPlaces(0) is not returning NULL, but should.");
	} else {
		Logger::info("Mass::getPlaces(0) passed.");
	}

	Logger::info("Done testing of Mass class.");
	failedTests += failures;
	return 0 == failures;
}

bool AllTests::runPlacesTests() {
	Logger::info("Beginning testing of Places class.");
	int failures = 0;

	int arg = 5;
	int nDims = 1;
	int size[] = { 10 };

	Places *p = Mass::createPlaces<TestPlace, TestState>(0, &arg, sizeof(int),
			nDims, size, 0);

	// test get dimensions
	if (p->getDimensions() != nDims) {
		++failures;
		Logger::error(
				"Places::getDimensions() returned incorrect number of dimensions.");
	} else {
		Logger::info("Places::getDimensions() passed.");
	}

	// test getHandle
	if (p->getHandle() != 0) {
		++failures;
		Logger::error("Places::getHandle() returned incorrect number.");
	} else {
		Logger::info("Places::getHandle() passed.");
	}

	// test index conversion functions
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

	// test dimensions storage
	if (!arrayequals(p->size(), size, nDims)) {
		++failures;
		Logger::error("Places::size() returned incorrect dimensions vector.");
	} else {
		Logger::info("Places::size() passed.");
	}

	// test place retreival and that they are set correctly on host
	Place ** myPlaces = p->getElements();
	if (NULL == myPlaces) {
		++failures;
		Logger::error("Places::getElements() returned NULL incorrectly.");
	} else {
		bool allOk = true;
		for (int i = 0; i < p->getNumPlaces(); ++i) {
			if (NULL == myPlaces[i]) {
				++failures;
				allOk = false;
				Logger::error("Places::myPlaces[i] is NULL incorrectly.");
				continue;
			}

			int retVal = *((int*) myPlaces[i]->getMessage());
			if (retVal != arg) {
				++failures;
				allOk = false;
				Logger::error("Place did not initialize with arg correctly.");
			}
		}

		if (allOk) {
			Logger::info("Place constructor set argument correctly.");
		}
	}

	// run callAll test
	Logger::info("Executing SET_TO_ONE test.");
	p->callAll(TestPlace::SET_TO_ONE);
	myPlaces = p->getElements();
	for (int i = 0; i < p->getNumPlaces(); ++i) {
		int val = *((int*) myPlaces[i]->getMessage());
		if (val != 1) {
			++failures;
			Logger::error(
					"Places.callAll(SET_TO_ONE) did not work. Value[%d] = %d",
					i, val);
			break;
		}
	}

	// run callAll test with an arg
	arg = 127;
	Logger::info("Executing SET_TO_ARG test where arg = %d", arg);
	p->callAll(TestPlace::SET_TO_ARG, &arg,sizeof(int));
	myPlaces = p->getElements();
	for (int i = 0; i < p->getNumPlaces(); ++i) {
		int val = *((int*) myPlaces[i]->getMessage());
		if (val != arg) {
			++failures;
			Logger::error(
					"Places.callAll(SET_TO_ARG) did not work. Value[%d] = %d",
					i, val);
			break;
		}
	}

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

