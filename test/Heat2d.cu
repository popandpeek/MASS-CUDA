/*
 *  @file Heat2d.cpp
 *  @author Nate Hart
 *	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#include "Heat2d.h"
#include <ctime> // clock_t
#include <iostream>
#include <math.h>     // floor
#include <sstream>     // ostringstream
#include <vector>

#include "../src/Mass.h"
#include "../src/Places.h"
#include "../src/Logger.h"
#include "Metal.h"
#include "Timer.h"

double a = 1.0;  // heat speed
double dt = 1.0; // time quantum
double dd = 2.0; // change in system

using namespace std;
using namespace mass;

Heat2d::Heat2d() {

}
Heat2d::~Heat2d() {

}

void Heat2d::displayResults(Places *places, int time, int *placesSize) {
	ostringstream ss;

	ss << "time = " << time << "\n";
	Place ** retVals = places->getElements();
	int indices[2];
	for (int x = 0; x < placesSize[0]; x++) {
		indices[0] = x;
		for (int y = 0; y < placesSize[1]; y++) {
			indices[1] = y;
			int rmi = places->getRowMajorIdx(indices);
			double temp = *((double*) retVals[rmi]->getMessage());
			ss << floor(temp / 2) << " ";
		}

		ss << "\n";
	}
	ss << "\n";
	Logger::print(ss.str());
}

void Heat2d::runMain() {

	string *arguments = NULL;
	int nGpu = 1; // # processes
	int nDims = 2;
	int placesSize[] = { 250, 250 };
	int max_time = 3000;
	int heat_time = 2700;
	int interval = 0;

	// start a process at each computing node
	Mass::init(arguments, nGpu);

	// distribute places over computing nodes
	double r = a * dt / (dd * dd);
	Places *places = Mass::createPlaces<Metal, MetalState>(0, &r,
			sizeof(double), nDims, placesSize, 0);

	// create neighborhood
	vector<int*> neighbors;
	int north[2] = { 0, 1 };
	neighbors.push_back(north);
	int east[2] = { 1, 0 };
	neighbors.push_back(east);
	int south[2] = { 0, -1 };
	neighbors.push_back(south);
	int west[2] = { -1, 0 };
	neighbors.push_back(west);

	// start a timer
	Timer time;
	time.start();

	// simulate heat diffusion in parallel
	for (int time = 0; time < max_time; time++) {

		if (time < heat_time)
			places->callAll(Metal::APPLY_HEAT);

		// display intermediate results
		if (interval != 0 && (time % interval == 0 || time == max_time - 1))
			displayResults(places, time, placesSize);

		places->exchangeAll(Metal::EXCHANGE, &neighbors);
		places->callAll(Metal::EULER_METHOD);
	}

	// finish the timer
	cout << "Elapsed time = " << time.lap() / 10000 / 100.0 << " seconds."
			<< endl;
	Logger::info("Elapsed time = %.2f seconds.",time.lap() / 10000 / 100.0 );

	// terminate the processes
	Mass::finish();
}

