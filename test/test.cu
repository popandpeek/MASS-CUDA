#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "../src/Mass.h"
#include "../src/Logger.h"

#include "Metal.h"
#include "Timer.h"

#include "../test_agents/SugarPlace.h"
#include "../test_agents/SugarPlaceState.h"
#include "../test_agents/Ant.h"
#include "../test_agents/AntState.h"


using namespace std;
using namespace mass;

static const int maxMetabolism = 4;
static const int maxInitAgentSugar = 10;

bool checkResults(Places *places, int time, int *placesSize) {
    bool correctResult = true;
    ostringstream ss;
    
    double targetResults[10][10] = {
        {0, 0, 1, 3, 4, 3, 1, 0, 0, 0},
        {0, 0, 1, 2, 3, 2, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

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
            double temp = ((Metal*) retVals[rmi])->getTemp();
            double roundedTemp = floor(temp / 2);
            ss << floor(temp / 2) << " ";
            if (roundedTemp != targetResults[row][col]) {
                correctResult = false;
            }
        }
        ss << "\n";
    }
    ss << "\n";

    if (!correctResult) {
        throw MassException("Incorrect simulation results: \n" + ss.str());
    } else {
        return true;
    }
}

bool checkSugarScapeResults(Places *places, int time, int *placesSize) {
    bool correctResult = true;
    ostringstream ss, agents;

    int targetSugar[10][10] = {
        {0, 0, 0, 1, 1, 1, 2, 1, 1, 1 },
        {0, 0, 0, 1, 2, 2, 1, 0, 2, 1 },
        {0, 0, 1, 1, 2, 3, 3, 3, 2, 1 },
        {0, 1, 1, 2, 2, 3, 3, 2, 1, 0 },
        {1, 2, 2, 2, 1, 0, 3, 3, 2, 1 },
        {1, 2, 2, 1, 0, 2, 2, 2, 2, 1 },
        {2, 2, 2, 1, 0, 1, 0, 1, 0, 1 },
        {1, 2, 1, 0, 3, 0, 1, 0, 0, 0 },
        {1, 2, 2, 2, 2, 2, 0, 0, 0, 0 },
        {1, 1, 1, 1, 0, 1, 1, 0, 0, 0 }
    };

    int targetAgentDistribution[10][10] ={
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
        {0, 0, 0, 0, 1, 0, 1, 0, 1, 0 },
        {0, 0, 0, 1, 0, 1, 0, 1, 0, 0 },
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }
    };

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
            if (curSugar != targetSugar[row][col]) {
                correctResult = false;
            }
            agents << n_agents << " ";
            if (n_agents != targetAgentDistribution[row][col]) {
                correctResult = false;
            }
        }
    }
    ss << "\n";
    agents << "\n";

    //TODO: compare with data from agent array

    if (!correctResult) {
        throw MassException("Incorrect simulation results: \n" + ss.str() + agents.str());
    } else {
        return true;
    }
}

void runMassSimTest(int size, int max_time, int heat_time) {

    string *arguments = NULL;
    int nDims = 2;
    int placesSize[] = { size, size };

    Mass::init(arguments);

    // initialization parameters
    double a = 1.0;  // heat speed
    double dt = 1.0; // time quantum
    double dd = 2.0; // change in system
    double r = a * dt / (dd * dd);

    // initialize places
    Places *places = Mass::createPlaces<Metal, MetalState>(0 /*handle*/, &r,
            sizeof(double), nDims, placesSize);

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
        places->exchangeAll(&neighbors);
        places->callAll(Metal::EULER_METHOD);
    }

    if (checkResults(places, time, placesSize)) {
        Logger::print("Intergation test successful!\n");
    }
    Mass::finish();
}

void runMassSimTestImproved(int size, int max_time, int heat_time) {

    string *arguments = NULL;
    int nDims = 2;
    int placesSize[] = { size, size };

    Mass::init(arguments);

    // initialization parameters
    double a = 1.0;  // heat speed
    double dt = 1.0; // time quantum
    double dd = 2.0; // change in system
    double r = a * dt / (dd * dd);

    // initialize places
    Places *places = Mass::createPlaces<Metal, MetalState>(0 /*handle*/, &r,
            sizeof(double), nDims, placesSize);

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
        places->exchangeAll(&neighbors, Metal::EULER_METHOD, NULL /*argument*/, 0 /*argSize*/);
    }

    if (checkResults(places, time, placesSize)) {
        Logger::print("Intergation test for improved library successful!\n");
    }
    Mass::finish();
}

void runSugarScapeTest(int size, int max_time) {
    string *arguments = NULL;
    int nDims = 2;
    int placesSize[] = { size, size };
    int nAgents = size*size / 5;

    Mass::init(arguments);

    // initialize places and agents
    Places *places = Mass::createPlaces<SugarPlace, SugarPlaceState>(0 /*handle*/, NULL /*arguments*/,
            sizeof(double), nDims, placesSize);
    places->callAll(SugarPlace::SET_SUGAR); //set proper initial amounts of sugar

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

    int t = 0;
    for (; t < max_time; t++) {

        places->callAll(SugarPlace::INC_SUGAR_AND_POLLUTION);

        places->exchangeAll(&neighbors, SugarPlace::AVE_POLLUTIONS, NULL /*argument*/, 0 /*argSize*/);
        places->callAll(SugarPlace::UPDATE_POLLUTION_WITH_AVERAGE);

        places->exchangeAll(&migrationDestinations, SugarPlace::FIND_MIGRATION_DESTINATION, NULL /*argument*/, 0 /*argSize*/); 
        
        agents->callAll(Ant::MIGRATE);

        agents->manageAll();

        agents->callAll(Ant::METABOLIZE);
        agents->manageAll();
    }

    if (checkSugarScapeResults(places, t, placesSize)) {
        Logger::print("Intergation test for library with dynamic agents is successful!\n");
    }

    // terminate the processes
    Mass::finish();
}

int main() {
    Logger::setLogFile("test_program_logs.txt");

    int size = 10;
    int max_time = 5;
    int heat_time = 4;

    Logger::print("Running intergation test using Heat2D program\n");

    runMassSimTest(size, max_time, heat_time);
    runMassSimTestImproved(size, max_time, heat_time);

    runSugarScapeTest(size, max_time);

    return 0;
}
