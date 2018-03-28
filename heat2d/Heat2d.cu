/*
 *  @file Heat2d.cpp
 	
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
#include "../src/Logger.h"
#include "Metal.h"
#include "Timer.h"

static const int WORK_SIZE = 32;
static const double a = 1.0;  // heat speed
static const double dt = 1.0; // time quantum
static const double dd = 2.0; // change in system

using namespace std;
using namespace mass;

Heat2d::Heat2d() {

}
Heat2d::~Heat2d() {

}

void Heat2d::displayResults(Places *places, int time, int *placesSize) {
	Logger::debug("Entering Heat2d::displayResults");
	ostringstream ss;

	ss << "time = " << time << "\n";
	Place ** retVals = places->getElements(); //refreshes places here
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
			double temp = ((Metal*) retVals[rmi])->getTemp();
			ss << floor(temp / 2) << " ";
		}

		ss << "\n";
	}
	ss << "\n";
	Logger::print(ss.str());
}

void Heat2d::runHostSim(int size, int max_time, int heat_time, int interval) {
	Logger::debug("Starting CPU simulation\n");
	double r = a * dt / (dd * dd);

	// create a space
	double ***z = new double**[2];
	for (int p = 0; p < 2; p++) {
		z[p] = new double*[size];
		for (int x = 0; x < size; x++) {
			z[p][x] = new double[size];
			for (int y = 0; y < size; y++) {
				z[p][x][y] = 0.0; // no heat or cold
			}
		}
	}

	// start a timer
	Timer timer;
	timer.start();

	// simulate heat diffusion
	int t = 0;
	int p;
	for (; t < max_time; t++) {
		p = t % 2; // p = 0 or 1: indicates the phase

		// two left-most and two right-most columns are identical
		for (int y = 0; y < size; y++) {
			z[p][0][y] = z[p][1][y];
			z[p][size - 1][y] = z[p][size - 2][y];
		}

		// two upper and lower rows are identical
		for (int x = 0; x < size; x++) {
			z[p][x][0] = z[p][x][1];
			z[p][x][size - 1] = z[p][x][size - 2];
		}

		// keep heating the bottom until t < heat_time
		if (t < heat_time) {
			for (int x = size / 3; x < size / 3 * 2; x++)
				z[p][x][0] = 19.0; // heat
		}

		// display intermediate results
		if (interval != 0 && (t % interval == 0 || t == max_time - 1)) {
			ostringstream ss;
			ss << "time = " << t << "\n";
			for (int y = 0; y < size; y++) {
				for (int x = 0; x < size; x++)
					ss << floor(z[p][x][y] / 2) << " ";
				ss << "\n";
			}
			ss << "\n";
			Logger::print(ss.str());
		}

		// perform forward Euler method
		int p2 = (p + 1) % 2;
		for (int x = 1; x < size - 1; x++)
			for (int y = 1; y < size - 1; y++)
				z[p2][x][y] = z[p][x][y]
						+ r * (z[p][x + 1][y] - 2 * z[p][x][y] + z[p][x - 1][y])
						+ r * (z[p][x][y + 1] - 2 * z[p][x][y] + z[p][x][y - 1]);

	} // end of simulation

	// finish the timer
//	cerr << "Elapsed time = " << timer.lap() / 10000 / 100.0 << endl;
//	Logger::info("Elapsed time on CPU = %.2f seconds.",
//			timer.lap() / 10000 / 100.0);
	Logger::print("CPU time: %d\n",timer.lap());


	if (size < 80) {
		ostringstream ss;
		ss << "time = " << t << "\n";
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++)
				ss << floor(z[p][x][y] / 2) << " ";
			ss << "\n";
		}
		ss << "\n";
		Logger::print(ss.str());
	}

	// clean up memory
	for (int p = 0; p < 2; p++) {
		for (int x = 0; x < size; x++) {
			delete[] z[p][x];
		}
		delete[] z[p];
	}
	delete[] z;
}

void Heat2d::runMassSim(int size, int max_time, int heat_time, int interval) {
	Logger::debug("Starting MASS CUDA simulation\n");

	int nDims = 2;
	int placesSize[] = { size, size };

	// start a process at each computing node
	Mass::init();
	Logger::debug("Finished Mass::init\n");

	// initialization parameters
	double r = a * dt / (dd * dd);

	// initialize places
	Places *places = Mass::createPlaces<Metal, MetalState>(0 /*handle*/, &r,
			sizeof(double), nDims, placesSize);
	Logger::debug("Finished Mass::createPlaces\n");

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
	Timer timer;
	timer.start();

	int time = 0;
	for (; time < max_time; time++) {

		if (time < heat_time) {
			places->callAll(Metal::APPLY_HEAT);
		}

		// display intermediate results
		if (interval != 0 && (time % interval == 0 || time == max_time - 1)) {
			displayResults(places, time, placesSize);
		}

		places->exchangeAll(&neighbors, Metal::EULER_METHOD, NULL /*argument*/, 0 /*argSize*/);
	}

	Logger::print("MASS time %d\n",timer.lap());

	if (placesSize[0] < 80) {
		displayResults(places, time, placesSize);
	}
	// terminate the processes
	Mass::finish();
}


/*
 * Begin implementation of custom CUDA implementation of HEAT2D
 */
__global__ void setCold(double *dest, double *src, int size) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < size * size) {
		dest[idx] = 0.0;
		src[idx] = 0.0;
	}
}

__device__ bool isLeftEdge(int idx, int size) {
	return idx % size == 0;
}

__device__ bool isRightEdge(int idx, int size) {
	return idx % size == size - 1;
}

__device__ bool isTopEdge(int idx, int size) {
	return idx < size;
}

__device__ bool isBottomEdge(int idx, int size) {
	return idx >= size * size - size;
}

__global__ void setEdges(double *dest, double *src, int size, int t,
		int heat_time, double r) {
	int idx = getGlobalIdx_1D_1D();

	if (idx < size * size) {

		// apply heat to top row
		if (idx >= size / 3 && idx < size / 3 * 2 && t < heat_time) {
			src[idx] = 19.0; // heat
			return;
		}
		// two left-most and two right-most columns are identical
		if (isLeftEdge(idx, size)) {
			src[idx] = src[idx + 1];
			return;
		}
		if (isRightEdge(idx, size)) {
			src[idx] = src[idx - 1];
			return;
		}

		// two upper and lower rows are identical
		if (isTopEdge(idx, size)) {
			src[idx] = src[idx + size];
			return;
		}
		if (isBottomEdge(idx, size)) {
			src[idx] = src[idx - size];
			return;
		}
	}
}

__global__ void euler(double *dest, double *src, int size, int t, int heat_time,
		double r) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
	int idx = threadId;

	// perform forward Euler method
	if (idx > size && idx < size * size - size) {
		double tmp = src[idx];
		dest[idx] = tmp + r * (src[idx + 1] - 2 * tmp + src[idx - 1])
				+ r * (src[idx + size] - 2 * tmp + src[idx - size]);
	}

}

void Heat2d::runDeviceSim(int size, int max_time, int heat_time, int interval) {
	Logger::debug("Starting GPU simulation\n");
	double r = a * dt / (dd * dd);

	// create a space
	double *z = new double[size * size];
	double *dest, *src;
	int nBytes = sizeof(double) * size * size;
	CATCH(cudaMalloc((void** ) &dest, nBytes));
	CATCH(cudaMalloc((void** ) &src, nBytes));

	int gridWidth = (size * size - 1) / WORK_SIZE + 1;
	int threadWidth = (size * size - 1) / gridWidth + 1;

	dim3 gridDim(gridWidth);
	dim3 threadDim(threadWidth);

	setCold<<<gridDim, threadDim>>>(dest, src, size);
	CHECK();

	// start a timer
	Timer time;
	time.start();

	// simulate heat diffusion
	int t = 0;
	for (; t < max_time; t++) {
		setEdges<<<gridDim, threadDim>>>(dest, src, size, t, heat_time, r);
		CHECK();

		// display intermediate results
		if (interval != 0 && (t % interval == 0 || t == max_time - 1)) {
			CATCH(cudaMemcpy(z, src, nBytes, D2H));
			ostringstream ss;
			ss << "time = " << t << "\n";
			for (int y = 0; y < size; y++) {
				for (int x = 0; x < size; x++)
					ss << floor(z[(y % size) * size + x] / 2) << " ";
				ss << "\n";
			}
			ss << "\n";
			Logger::print(ss.str());
		}

		euler<<<gridDim, threadDim>>>(dest, src, size, t, heat_time, r);
		CHECK();

		double *swap = dest;
		dest = src;
		src = swap;
	} // end of simulation

	// finish the timer
//	Logger::info("Elapsed time on GPU = %.2f", time.lap() / 10000 / 100.0);
	Logger::print("GPU time: %d\n",time.lap());

	if (size < 80) {
		CATCH(cudaMemcpy(z, dest, nBytes, D2H));
		ostringstream ss;
		ss << "time = " << t << "\n";
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++)
				ss << floor(z[(y % size) * size + x] / 2) << " ";
			ss << "\n";
		}
		ss << "\n";
		Logger::print(ss.str());
	}
	delete[] z;
}

