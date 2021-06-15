#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "../src/Mass.h"
#include "../src/Logger.h"

#include "Heat2d.h"

using namespace std;
using namespace mass;

int main() {
	// test logging
	Logger::setLogFile("heat2d_results.txt");

	// const int nRuns = 1; // number of times to run each test
	// const int nSizes = 11;
	// int size[nSizes] = { 5, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
	// int max_time = 3000;
	// int heat_time = 2700;
	// int interval = 0;

	// For debigging can use the following settings:
	const int nRuns = 1;
	const int nSizes = 1;
	int size[nSizes] = { 100 };
	int max_time = 10;
	int heat_time = 8;
	int interval = 1;

	Logger::print("Size,CPU,GPU,MASS\n");
	Heat2d heat;
	for (int i = 0; i < nSizes; ++i) {
		Logger::info(
			"Running Heat 2D with params:\n\tsize = %d\n\ttime = %d\n\theat_time = %d\n\tinterval = %d",
			size[i], max_time, heat_time, interval);

		for (int run = 0; run < nRuns; ++run) {
			Logger::print("%d,",size[i]);
			// heat.runHostSim(size[i], max_time, heat_time, interval);
			// heat.runDeviceSim(size[i], max_time, heat_time, interval);
			heat.runMassSim(size[i], max_time, heat_time, interval);
		}
	}

	return 0;
}
