/**
 *  @file main.cpp
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "tests/AllTests.h"
#include "Mass.h"

using namespace std;
using namespace mass;

int main() {

	AllTests tests;
	stringstream ss;

	// test logging
	Mass::setLogFile("test_results.txt");
	Mass::log("Logging in test file successfully.");

	Mass::init();

	Mass::log("Mass::init() successful.");

	Mass::log("Beginning testing of Mass class.");
	if (!tests.runMassTests()) {
		ss << "\tMass Tests\n";
	}
	Mass::log("Done testing Mass class.");

	Mass::log("Beginning testing of Dispatcher class.");
	if (!tests.runDispatcherTests()) {
		ss << "\tDispatcher Tests\n";
	}

	if (!tests.runPlacesPartitionTests()) {
		ss << "\tPlacesPartition Tests\n";
	}

	if (!tests.runPlacesTests()) {
		ss << "\tPlaces Tests\n";
	}

	if (!tests.runAgentsPartitionTests()) {
		ss << "\tAgentsPartition Tests\n";
	}

	if (!tests.runAgentsTests()) {
		ss << "\tAgents Tests\n";
	}


	Mass::log("Calling Mass::finish()");
	Mass::finish();

	Mass::log("Mass::finish() passed.");

	cout << "Tests finished. The following tests failed:\n" << ss.str() << "\n"
			<< "End failed tests." << endl;
	return 0;
}
