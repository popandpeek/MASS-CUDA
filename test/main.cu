#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "../src/Mass.h"
#include "../src/Logger.h"

#include "AllTests.h"
#include "Heat2d.h"

using namespace std;
using namespace mass;

int main() {
	// test logging
	Logger::setLogFile("test_results.txt");

//
//	AllTests tests;
//	stringstream ss;
//
//
//	int ngpu = 1;
//	Mass::init(NULL, ngpu);
//
//	Logger::info("Mass::init() successful.");
//
////	if(!tests.proofOfConcept()){
////		ss << "\tProof Of Concept Tests\n";
////	}
//
//	if (!tests.runMassTests()) {
//		ss << "\tMass Tests\n";
//	}
//
//	if (!tests.runPlacesTests()) {
//		ss << "\tPlaces Tests\n";
//	}
//
////
////	if (!tests.runDispatcherTests()) {
////		ss << "\tDispatcher Tests\n";
////	}
////
////	if (!tests.runPlacesPartitionTests()) {
////		ss << "\tPlacesPartition Tests\n";
////	}
////
////	if (!tests.runAgentsPartitionTests()) {
////		ss << "\tAgentsPartition Tests\n";
////	}
////
////	if (!tests.runAgentsTests()) {
////		ss << "\tAgents Tests\n";
////	}
////
//	Logger::info("Calling Mass::finish()");
//	Mass::finish();
////
//	Logger::info("Mass::finish() passed.");
//
////	cout << "All Tests finished. The following tests failed:\n" << ss.str()
////			<< "\n" << "End failed tests." << endl;
//	char buf[500];
//	strcpy(buf, ss.str().c_str());
//	if (tests.failedTests > 0) {
//		Logger::info("Tests finished. The following tests failed:\n%s\n", buf);
//	} else {
//		Logger::info("All tests passed.");
//	}

	Logger::info("Running Heat 2D");
	Heat2d heat;
	heat.runMain();

	return 0;
}
