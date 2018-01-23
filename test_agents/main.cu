#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "../src/Mass.h"
#include "../src/Logger.h"

#include "SugarScape.h"

using namespace std;
using namespace mass;

int main() {
	// test logging
	Logger::setLogFile("sugar_scape_test_results.txt");

	// const int nRuns = 3; // number of times to run each test
	// const int nSizes = 4;
	// int size[nSizes] = { 125, 250, 500, 1000};
	// int max_time = 3000;
	// int interval = 0;

	const int nRuns = 1; // number of times to run each test
	const int nSizes = 1;
	int size[nSizes] = {10};
	int max_time = 10;
	int interval = 1;

	Logger::print("Size,CPU,GPU,MASS\n");
	SugarScape sugarScape;
	for (int i = 0; i < nSizes; ++i) {
		Logger::info(
			"Running SugarScape with params:\n\tsize = %d\n\ttime = %d\n\tinterval = %d",
			size[i], max_time, interval);

		for (int run = 0; run < nRuns; ++run) {
			Logger::print("%d,",size[i]);
			// sugarScape.runHostSim(size[i], max_time, interval);
			// sugarScape.runDeviceSim(size[i], max_time, interval);
			sugarScape.runMassSim(size[i], max_time, interval);
		}
	}

	return 0;
}
