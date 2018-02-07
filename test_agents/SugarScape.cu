
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

static const int maxMetabolism = 4;
static const int maxInitAgentSugar = 10;

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
	int nAgents = agents->getNumAgents();
	for (int i =0; i< nAgents; i++) {
		if (retVals[i] -> isAlive()) {
			int placeIdx = retVals[i] -> getPlaceIndex();
			int agentSugar = ((AntState*)(retVals[i]->getState()))->agentSugar;
			int agentMetabolism = ((AntState*)(retVals[i]->getState()))->agentMetabolism;
			ss << "Agent[" << i << "] at location " << placeIdx << ", agentSugar = " << agentSugar << ", agentMetabolism = " << agentMetabolism << endl;
		}
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


	//initialize agents:
	Agents *agents = Mass::createAgents<Ant, AntState> (1 /*handle*/, NULL /*arguments*/,
			sizeof(double), nAgents, 0 /*placesHandle*/);

	//create an array of random agentSugar and agentMetabolism values
	int agentSugarArray[nAgents];
	int agentMetabolismArray[nAgents];
	for (int i=0; i<nAgents; i++) {
		agentSugarArray[i] = rand() % maxInitAgentSugar +1;
		agentMetabolismArray[i] = rand() % maxMetabolism +1;
	}

	//set proper initial amounts of sugar and metabolism for agents
	agents->callAll(Ant::SET_INIT_SUGAR, agentSugarArray, sizeof(int) * nAgents);
	agents->callAll(Ant::SET_INIT_METABOLISM, agentMetabolismArray, sizeof(int) * nAgents);

	// create neighborhood for average pollution calculations
	// TODO: move from x,y coordinates to 1D coordinates, because we use 1D averywhere else anyway
	vector<int*> neighbors;
	int top[2] = { 0, 1 };
	neighbors.push_back(top);
	int right[2] = { 1, 0 };
	neighbors.push_back(right);
	int bottom[2] = { 0, -1 };
	neighbors.push_back(bottom);
	int left[2] = { -1, 0 };
	neighbors.push_back(left);

	// create a vector of possible target destination places for an ant
    vector<int*> migrationDestinations;
    //TODO: modify all versions of the program to include left and up as well
    for(int i=1;i<=maxVisible; i++ ) //going right
    {
        int *hDest = new int[2];
        hDest[0] = i;
        hDest[1] = 0;
        migrationDestinations.push_back(hDest);
    }
    for(int i=1;i<=maxVisible; i++ ) //going up
    {
        int *vDest = new int[2];
        vDest[0] = 0;
        vDest[1] = i;
        migrationDestinations.push_back(vDest);
    }

	// start a timer
	Timer timer;
	timer.start();

	int t = 0;
	for (; t < max_time; t++) {

		places->callAll(SugarPlace::INC_SUGAR_AND_POLLUTION);

		places->exchangeAll(&neighbors, SugarPlace::AVE_POLLUTIONS, NULL /*argument*/, 0 /*argSize*/);
		places->callAll(SugarPlace::UPDATE_POLLUTION_WITH_AVERAGE);

		places -> callAll(SugarPlace::IDENTIFY_IF_GOOD_FOR_MIGRATION);
		places->exchangeAll(&migrationDestinations, SugarPlace::FIND_MIGRATION_DESTINATION, NULL /*argument*/, 0 /*argSize*/); 
		
		agents->callAll(Ant::MIGRATE);

		agents->manageAll();

		agents->callAll(Ant::METABOLIZE);
		agents->manageAll();

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

