#include <ctime> // clock_t
#include <iostream>
#include <math.h>     // floor
#include <sstream>     // ostringstream
#include <vector>

#include "../src/Mass.h"
#include "../src/Logger.h"
#include "BrainGrid.h"
#include "NeuronPlace.h"
#include "GrowingEnd.h"
#include "Timer.h"
#include "NeuronPlaceState.h"
#include "BrainGridConstants.h"


using namespace std;
using namespace mass;

BrainGrid::BrainGrid() {

}

BrainGrid::~BrainGrid() {

}

void BrainGrid::displaySoma(Places *places, int time, int *placesSize) {
	Logger::debug("Entering BrainGrid::displaySoma");
	ostringstream ss;
	ostringstream agents;

	ss << "time = " << time << "\n";
	Place** retVals = places->getElements();
	Logger::debug("BrainGrid: Returns from places->getElements()");
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

			const char* soma = (((NeuronPlace*)retVals[rmi])->getPlaceType() == BrainGridConstants::SOMA) ? "X" : "O";
            int n_agents = retVals[rmi]->getAgentPopulation();
			ss << soma << " ";
			agents << n_agents << " ";
			//Logger::debug("BrainGrid: Exits inner loop: Count = %d out of %d.", col, placesSize[1]);
		}
		
		//Logger::debug("BrainGrid: Exits outer loop: Count = %d out of %d.", row, placesSize[0]);
		ss << "\n";
		agents << "\n";
	}
	ss << "\n";
	agents << "\n";
	Logger::print(ss.str());
	Logger::print(agents.str());
	delete[] retVals;
}

void BrainGrid::displayConnectedSomas(Places *places, int time, int *placesSize) {
	Logger::debug("Entering BrainGrid::displayConnectedSoma");
	ostringstream ss;
	ostringstream agents;

	ss << "time = " << time << "\n";
	Place** retVals = places->getElements();
	Logger::debug("BrainGrid: Returns from places->getElements()");
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

			const char* soma = (((NeuronPlace*)retVals[rmi])->isOccupied()) ? "X" : "_";
            int n_agents = retVals[rmi]->getAgentPopulation();
			ss << soma << " ";
			agents << n_agents << " ";
			//Logger::debug("BrainGrid: Exits inner loop: Count = %d out of %d.", col, placesSize[1]);
		}
		
		//Logger::debug("BrainGrid: Exits outer loop: Count = %d out of %d.", row, placesSize[0]);
		ss << "\n";
		agents << "\n";
	}
	ss << "\n";
	agents << "\n";
	Logger::print(ss.str());
	// Logger::print(agents.str());
	delete[] retVals;
}

void BrainGrid::displayAgents(Agents* agents, int time) {
	Logger::debug("Entering BrainGrid::displayAgents");
	ostringstream ss;
	ss << "time = " << time << "\n";
	mass::Agent** retVals = agents->getElements();
	int nAgentObjects = agents->getNumAgentObjects(); 
    Logger::debug("**** displayAgents: nAgentObjects = %d", nAgentObjects);
	for (int i =0; i< nAgentObjects; i++) {
		if (retVals[i] -> isAlive()) {
			int placeIdx = retVals[i] -> getPlaceIndex();
            int endSoma = ((GrowingEnd*)(retVals[i]))->getSomaIndex();
			int agentSignal = ((GrowingEnd*)(retVals[i]))->getSignal();
			int endType = ((GrowingEnd*)(retVals[i]))->getAgentType();
            bool atSoma = (retVals[i] -> getPlaceIndex() == ((GrowingEnd*)(retVals[i]))->getSomaIndex());
			ss << "Agent[" << i << "] of type " << endType << " at location " << placeIdx << ", SOMA location = " << endSoma <<  ", at SOMA = " << atSoma << endl;
        }
	}

	Logger::print(ss.str());
}

void BrainGrid::runMassSim(int size, int max_time, int interval) {
    Logger::debug("Starting MASS CUDA BrainGrid simulation\n");
    
    int nDims = 2;
	int placesSize[] = { size, size };
	Logger::debug("placesSize[0] = %d, placesSize[1] = %d", placesSize[0], placesSize[1]);
    
    // start the MASS CUDA library processes
	Mass::init();

	// initialize places
	Places *neurons = Mass::createPlaces<NeuronPlace, NeuronPlaceState>(0 /*handle*/, NULL /*arguments*/,
			sizeof(double), nDims, placesSize);

    int* mx_time = new int[size * size];
    for (int i = 0; i < size * size; ++i) {
        mx_time[i] = max_time;
    }
    neurons->callAll(NeuronPlace::SET_TIME, mx_time, sizeof(int) * size * size);
    delete mx_time;

    // Initializes neurons as empty space or containing a neuron and the type of neuron - excitatory, inhibitory, or neutral
    int* randos = Mass::getRandomNumbers(size * size, 100);
    int neuronLocations[size*size]{0};
    int numNeurons = 0;
    
    for (int i = 0; i < size*size; ++i) {
        if (randos[i] >= BrainGridConstants::SPACE) {
            neuronLocations[i] = BrainGridConstants::SOMA;
            numNeurons++;
        }
    }

    randos = Mass::getRandomNumbers(size * size, 100);

    // Set neurons - SOMA or EMPTY
    neurons->callAll(NeuronPlace::INIT_NEURONS, neuronLocations, sizeof(int) * size * size);
    // set type of neuron signal
    neurons->callAll(NeuronPlace::SET_NEURON_SIGNAL_TYPE, randos, sizeof(int) * size * size);

    randos = Mass::getRandomNumbers(size * size, 100);
    neurons->callAll(NeuronPlace::SET_GROWTH_DIRECTIONS, randos, sizeof(int) * size * size);
    
    // Locations for Agent instantiation 
    int locations[numNeurons]{ 0 };
    int count = 0;
    for (int i = 0; i < size * size; ++i) {
        if (neuronLocations[i] == BrainGridConstants::SOMA) {
            locations[count] = i;
            count++;
        }
    }

    // create neighborhood for Axon and Dendrite potential branch and growth destinations
	vector<int*> neighbors;
	int north[2] = { 0, 1 };
    neighbors.push_back(north);
    int northEast[2] = { 1, 1 };
    neighbors.push_back(northEast);
	int east[2] = { 1, 0 };
    neighbors.push_back(east);
    int southEast[2] = { 1, -1 };
    neighbors.push_back(southEast);
	int south[2] = { 0, -1 };
    neighbors.push_back(south);
    int southWest[2] = { -1, -1 };
    neighbors.push_back(southWest);
	int west[2] = { -1, 0 };
    neighbors.push_back(west);
    int northWest[2] = { -1, 1 };
    neighbors.push_back(northWest);

    // Sets spawn times for axon and all dendrites in each neuron with SOMA
    randos = Mass::getRandomNumbers(size * size * MAX_NEIGHBORS, max_time);
    neurons->exchangeAll(&neighbors, NeuronPlace::SET_SPAWN_TIMES, randos, 
        sizeof(int) * size * size * MAX_NEIGHBORS);

    neurons->callAll(NeuronPlace::FIND_AXON_GROWTH_DESTINATIONS_FROM_SOMA);

    // One Agent for each AXON and DENDRITE for each Place with SOMA
	Agents *axons = Mass::createAgents<GrowingEnd, GrowingEndState> (1 /*handle*/, NULL /*arguments*/,
        0 /*argSize*/, numNeurons /*nAgents*/, 0 /*placesHandle*/, numNeurons * 2 /*maxAgents*/, locations /*placeIdxs*/);
    
    Agents *dendrites = Mass::createAgents<GrowingEnd, GrowingEndState> (2 /*handle*/, NULL /*arguments*/, 
        0 /*argSize*/, numNeurons /*nAgents*/, 0 /*placesHandle*/, numNeurons * (MAX_NEIGHBORS - 1) * 2 /*maxAgents*/, locations /*placeIdxs*/);
    
    // initialize agents
    axons->callAll(GrowingEnd::INIT_AXONS);
    
    // start a timer
	Timer timer;
    timer.start();
    
    int t = 0;
    
    // displaySoma(neurons, t, placesSize);
    // displayAgents(axons, t);
    // displayAgents(dendrites, t);

    for (; t < max_time; t++) {
        Logger::debug(" ******* BrainGrid Main Loop, iteration: %d *******\n", t);

        dendrites->callAll(GrowingEnd::INIT_DENDRITES);

        // If time to spawn, spawn
        axons->callAll(GrowingEnd::SPAWN_AXONS);
        dendrites->callAll(GrowingEnd::SPAWN_DENDRITES);

        dendrites->spawnAll();

        // Prepare new dendrites for growth
        neurons->callAll(NeuronPlace::FIND_DENDRITE_GROWTH_DESTINATIONS_FROM_SOMA);

        // Prepare for growth of Axons and Dendrites on SOMA (migrate)
        axons->callAll(GrowingEnd::GROW_AXON_SOMA);
        dendrites->callAll(GrowingEnd::GROW_DENDRITE_SOMA);

        // Migrate Axons, Synapses, and Dendrites from SOMA
        axons->migrateAll();
        dendrites->migrateAll();

        // Check if Axons switch to synapses
        randos = Mass::getRandomNumbers(size*size, 100);
        axons->callAll(GrowingEnd::AXON_TO_SYNAPSE, randos, sizeof(int) * size*size);

        // If a synapse and dendrite meet, link them 
        neurons->callAll(NeuronPlace::MAKE_CONNECTIONS);
        neurons->callAll(NeuronPlace::REMOVE_MARKED_AGENTS_FROM_PLACE);

        axons->terminateAll();
        dendrites->terminateAll();

        neurons->callAll(NeuronPlace::FIND_GROWTH_DESTINATIONS_OUTSIDE_SOMA);

        randos = Mass::getRandomNumbers(size*size, 100);
        axons->callAll(GrowingEnd::GROW_AXONS_OUTSIDE_SOMA, randos, size*size * sizeof(int));

        // Grow Synapses not on SOMA's
        randos = Mass::getRandomNumbers(axons->getMaxAgents(), 100);
        axons->callAll(GrowingEnd::GROW_SYNAPSE, randos, axons->getMaxAgents() * sizeof(int));

        // Grow Dendrites not on SOMA's
        randos = Mass::getRandomNumbers(dendrites->getMaxAgents(), 100);
        dendrites->callAll(GrowingEnd::GROW_DENDRITE, randos, dendrites->getMaxAgents() * sizeof(int));
        
        neurons->callAll(NeuronPlace::REMOVE_MARKED_AGENTS_FROM_PLACE);
        axons->terminateAll();
        axons->migrateAll();
        dendrites->terminateAll();
        dendrites->migrateAll();    

        neurons->callAll(NeuronPlace::MAKE_CONNECTIONS);
        neurons->callAll(NeuronPlace::REMOVE_MARKED_AGENTS_FROM_PLACE);

        axons->terminateAll();
        dendrites->terminateAll();

        neurons->callAll(NeuronPlace::FIND_BRANCH_DESTINATIONS_OUTSIDE_SOMA);

        randos = Mass::getRandomNumbers(axons->getMaxAgents(), 100);
        axons->callAll(GrowingEnd::BRANCH_SYNAPSES, randos, axons->getMaxAgents() * sizeof(int));

        randos = Mass::getRandomNumbers(dendrites->getMaxAgents(), 100);
        dendrites->callAll(GrowingEnd::BRANCH_DENDRITES, randos, dendrites->getMaxAgents() * sizeof(int));

        axons->spawnAll();
        dendrites->spawnAll();

        axons->callAll(GrowingEnd::SET_SYNAPSES);
        dendrites->callAll(GrowingEnd::SET_DENDRITES);  

        axons->callAll(GrowingEnd::GROW_BRANCHES);
        dendrites->callAll(GrowingEnd::GROW_BRANCHES);
        neurons->callAll(NeuronPlace::REMOVE_MARKED_AGENTS_FROM_PLACE);
        
        axons->terminateAll();
        axons->migrateAll();
        dendrites->terminateAll();
        dendrites->migrateAll();      

        neurons->callAll(NeuronPlace::MAKE_CONNECTIONS);
        neurons->callAll(NeuronPlace::REMOVE_MARKED_AGENTS_FROM_PLACE);
        
        // // 1. Axon Agent travels to its SOMA
        // axons->callAll(GrowingEnd::SOMA_TRAVEL);

        // axons->migrateAll();

        // // Collect and transmit signals
        // // 2. SOMAs create signal
        // randos = Mass::getRandomNumbers(neurons->getNumPlaces(), 100);
        // neurons->callAll(NeuronPlace::CREATE_SIGNAL, randos, neurons->getNumPlaces() * sizeof(int));

        // // 3. synapses on SOMA's get signal
        // axons->callAll(GrowingEnd::GET_SIGNAL);

        // // 4. synapses migrate to SOMA
        // axons->callAll(GrowingEnd::DENDRITE_SOMA_TRAVEL); 

        // axons->migrateAll();

        // // 5. dendrites place signal at SOMA
        // axons->callAll(GrowingEnd::SET_SOMA_SIGNAL);

        // // 6. SOMAs process received signals
        // neurons->callAll(NeuronPlace::PROCESS_SIGNALS);

        // // 7. Axons migrate to SOMA home
        // axons->callAll(GrowingEnd::SOMA_TRAVEL);

        // axons->migrateAll();

        // Update end of iter params
        neurons->callAll(NeuronPlace::UPDATE_ITERS);
        axons->callAll(GrowingEnd::UPDATE_ITERS);
        dendrites->callAll(GrowingEnd::UPDATE_ITERS);

        // display intermediate results
		if (interval != 0 && (t % interval == 0 || t == max_time - 1)) {
			displaySoma(neurons, t, placesSize);
            displayConnectedSomas(neurons, t, placesSize);
			displayAgents(axons, t);
            displayAgents(dendrites, t);
		}
    }

    Logger::print("MASS time %d\n",timer.lap());

    if (placesSize[0] < 80) {
		displaySoma(neurons, t, placesSize);
        displayConnectedSomas(neurons, t, placesSize);
		displayAgents(axons, t);
        displayAgents(dendrites, t);
	}
    
    // terminate the processes
	Mass::finish();
}