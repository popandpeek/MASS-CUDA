#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include "../src/Mass.h"
#include "../src/Logger.h"

#include "Metal.h"
#include "Timer.h"

using namespace std;
using namespace mass;

void checkResults(Places *places, int time, int *placesSize) {
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
            double temp = *((double*) retVals[rmi]->getMessage());
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
        Logger::print("Intergation test successful!\n");
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

    checkResults(places, time, placesSize);
    Mass::finish();
}

int main() {
    Logger::setLogFile("test_program_logs.txt");

    int size = 10;
    int max_time = 5;
    int heat_time = 4;

    Logger::print("Running intergation test using Heat2D program\n");

    runMassSimTest(size, max_time, heat_time);

    return 0;
}
