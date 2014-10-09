/**
 *  @file AllTests.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "AllTests.h"
#include "src/Mass.h"
#include "src/Places.h"

using namespace mass;

AllTests::AllTests() {
}

AllTests::~AllTests() {
}

bool AllTests::runDispatcherTests() {
//	Mass::log("Beginning testing of Dispatcher class.");
//	Mass::log("Done testing of Dispatcher class.");
	return true;
}

bool AllTests::runMassTests() {
	Mass::log("Beginning testing of Mass class.");
	int failures = 0;

	if (Mass::numAgentsInstances() != 0) {
		++failures;
		Mass::log(
				"Mass::numAgentsInstances() returning more than 0, but should be 0.");
	} else {
		Mass::log("Mass::numAgentsInstances() passed.");
	}

	if (Mass::numPlacesInstances() != 0) {
		++failures;
		Mass::log(
				"Mass::numPlacesInstances() returning more than 0, but should be 0.");
	} else {
		Mass::log("Mass::numPlacesInstances() passed.");
	}

	if (NULL != Mass::getAgents(0)) {
		++failures;
		Mass::log("Mass::getAgents(0) is not returning NULL, but should.");
	} else {
		Mass::log("Mass::getAgents(0) passed.");
	}

	if (NULL != Mass::getPlaces(0)) {
		++failures;
		Mass::log("Mass::getPlaces(0) is not returning NULL, but should.");
	} else {
		Mass::log("Mass::getPlaces(0) passed.");
	}

	Mass::log("Done testing of Mass class.");
	return 0 == failures;
}

bool AllTests::runPlacesTests() {
	Mass::log("Beginning testing of Places class.");

	int arg = 5;
	int size[] = {5};
	Places *p = new Places(0, "Debug/tests/TestPlace.d", &arg, sizeof(int),1,size,0);

	Mass::log("Done testing of Places class.");
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

