
#include "SugarScape.h"
#include <ctime> // clock_t
#include <iostream>
#include <math.h>     // floor
#include <sstream>     // ostringstream
#include <vector>

#include "../src/Mass.h"
#include "../src/Logger.h"
#include "SugarPlace.h"
#include "Timer.h"

using namespace std;
using namespace mass;

SugarScape::SugarScape() {

}
SugarScape::~SugarScape() {

}

void SugarScape::displaySugar(Places *places, int time, int *placesSize) {
	Logger::debug("Entering Heat2d::displayResults");
	ostringstream ss;

	ss << "time = " << time << "\n";
	Place ** retVals = places->getElements();
	int indices[2];
	for (int row = 0; row < placesSize[0]; row++) {
		indices[0] = row;
		for (int col = 0; col < placesSize[1]; col++) {
			indices[1] = col;
			int rmi = places->getRowMajorIdx(indices);
			if (rmi != (row % placesSize[0]) * placesSize[0] + col) {
				Logger::error("Row Major Index is incorrect: [%d][%d] != %d",
						row, col, rmi);
			}
			int curSugar = *((int*) retVals[rmi]->getMessage());
			ss << curSugar << " ";
		}

		ss << "\n";
	}
	ss << "\n";
	Logger::print(ss.str());
}

// void SugarScape::runHostSim(int size, int max_time, int interval) {
	
// }

void SugarScape::runMassSim(int size, int max_time, int interval) {
	Logger::debug("Starting MASS CUDA simulation\n");

	string *arguments = NULL;
	int nDims = 2;
	int placesSize[] = { size, size };

	// start a process at each computing node
	Mass::init(arguments);

	// initialization parameters
	// int nAgents = size*size / 5;

	// initialize places
	Places *places = Mass::createPlaces<SugarPlace, SugarPlaceState>(0 /*handle*/, NULL,
			sizeof(double), nDims, placesSize);
	Logger::debug("Finished Mass::createPlaces\n");

	// // create neighborhood
	// vector<int*> neighbors;
	// int north[2] = { 0, 1 };
	// neighbors.push_back(north);
	// int east[2] = { 1, 0 };
	// neighbors.push_back(east);
	// int south[2] = { 0, -1 };
	// neighbors.push_back(south);
	// int west[2] = { -1, 0 };
	// neighbors.push_back(west);

	// start a timer
	Timer timer;
	timer.start();

	int time = 0;
	for (; time < max_time; time++) {

	// 	if (time < heat_time) {
	// 		places->callAll(Metal::APPLY_HEAT);
	// 	}

	// 	// display intermediate results
	// 	if (interval != 0 && (time % interval == 0 || time == max_time - 1)) {
	// 		displaySugar(places, time, placesSize);
	// 	}

	// 	places->exchangeAll(&neighbors, Metal::EULER_METHOD, NULL /*argument*/, 0 /*argSize*/);
	}

	Logger::print("MASS time %d\n",timer.lap());

	if (placesSize[0] < 80) {
		displaySugar(places, time, placesSize);
	}
	// terminate the processes
	Mass::finish();
}

// void SugarScape::runDeviceSim(int size, int max_time, int interval) {

// }

