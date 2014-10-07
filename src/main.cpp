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

	if (!tests.runMassTests()) {
		ss << "\tMass Tests\n";
	}

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

	cout << "Tests finished. The following tests failed:\n" << ss.str() << "\n"
			<< "End failed tests." << endl;
	return 0;
}
