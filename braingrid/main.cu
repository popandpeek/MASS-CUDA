#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "../src/Mass.h"
#include "../src/Logger.h"

#include "BrainGrid.h"

using namespace std;
using namespace mass;

int main() {
	// test logging
    Logger::setLogFile("sugar_scape_results.txt");
 
    // Set-up run params
    const int nRuns = 1; // number of times to run each test
	const int nSizes = 1;
	int size[nSizes] = { 100 }; // 200, 300, 400, 500, 600, 700, 800, 900, 1000};
	int max_time = 100;
    int interval = 0;
    
    Logger::print("Size,CPU,GPU,MASS\n");
	BrainGrid brainGrid;
	for (int i = 0; i < nSizes; ++i) {
		Logger::info(
			"Running BrainGrid with params:\n\tsize = %d\n\ttime = %d\n\tinterval = %d",
			size[i], max_time, interval);

		for (int run = 0; run < nRuns; ++run) {
			Logger::print("%d,",size[i]);
			brainGrid.runMassSim(size[i], max_time, interval);
		}
	}

	return 0;

}