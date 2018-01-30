
#include "SugarScape.h"
#include <ctime> // clock_t
#include <iostream>
#include <math.h>     // floor
#include <sstream>     // ostringstream
#include <vector>

#include "../src/Mass.h"
#include "../src/Logger.h"
#include "SugarPlace.h"
#include "Ant.h"
#include "Timer.h"

using namespace std;
using namespace mass;

SugarScape::SugarScape() {

}
SugarScape::~SugarScape() {

}

void SugarScape::displaySugar(Places *places, int time, int *placesSize) {
	Logger::debug("Entering SugarScape::displayResults");
	ostringstream ss;
	ostringstream agents;

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
			int curSugar = ((SugarPlace*)retVals[rmi])->getCurSugar();
			int n_agents = retVals[rmi]->getAgentPopulation();
			ss << curSugar << " ";
			agents << n_agents << " ";
		}

		ss << "\n";
		agents << "\n";
	}
	ss << "\n";
	agents << "\n";
	Logger::print(ss.str());
	Logger::print(agents.str());
}

void SugarScape::displayAgents(Agents* agents, int time) {
	Logger::debug("Entering SugarScape::displayAgents");
	ostringstream ss;
	ss << "time = " << time << "\n";
	mass::Agent** retVals = agents->getElements();
	Logger::debug("SugarScape::displayAgents _____after getElements");
	int nAgents = agents->getNumAgents();
	Logger::debug("SugarScape::displayAgents _____after getNumAgents");
	for (int i =0; i< nAgents; i++) {
		int placeIdx = retVals[i] -> getPlaceIndex();
		ss << "Agent[" << i << "] at location " << placeIdx << endl;   //FAILS HERE!!! Segmentation fault
	}
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

	//initialization parameters
	int nAgents = size*size / 5;

	// initialize places
	Places *places = Mass::createPlaces<SugarPlace, SugarPlaceState>(0 /*handle*/, NULL /*arguments*/,
			sizeof(double), nDims, placesSize);

	places->callAll(SugarPlace::SET_SUGAR); //set proper initial amounts of sugar

	//initialize agents

	Agents *agents = Mass::createAgents<Ant, AntState> (1 /*handle*/, NULL /*arguments*/,
			sizeof(double), nAgents, 0 /*placesHandle*/);

	// create neighborhood
	vector<int*> neighbors;
	int top[2] = { 0, 1 };
	neighbors.push_back(top);
	int right[2] = { 1, 0 };
	neighbors.push_back(right);
	int bottom[2] = { 0, -1 };
	neighbors.push_back(bottom);
	int left[2] = { -1, 0 };
	neighbors.push_back(left);

	// start a timer
	Timer timer;
	timer.start();

	int t = 0;
	for (; t < max_time; t++) {

		places->callAll(SugarPlace::INC_SUGAR_AND_POLLUTION);

		places->exchangeAll(&neighbors, SugarPlace::AVE_POLLUTIONS, NULL /*argument*/, 0 /*argSize*/);

		places->callAll(SugarPlace::UPDATE_POLLUTION_WITH_AVERAGE);

		// display intermediate results
		if (interval != 0 && (t % interval == 0 || t == max_time - 1)) {
			displaySugar(places, t, placesSize);
			displayAgents(agents, t);
		}
	}

	Logger::print("MASS time %d\n",timer.lap());

	if (placesSize[0] < 80) {
		displaySugar(places, t, placesSize);
		displayAgents(agents, t);
	}
	// terminate the processes
	Mass::finish();
}

// void SugarScape::runDeviceSim(int size, int max_time, int interval) {

// }

