
#include "BrainGrid.h"
#include <ctime> // clock_t
#include <iostream>
#include <math.h>     // floor
#include <sstream>     // ostringstream
#include <vector>

#include "../src/Mass.h"
#include "../src/Logger.h"
#include "NeuronPlace.h"
#include "Axon.h"
#include "Dendrite.h"
#include "Timer.h"
#include "NeuronPlaceState.h"
#include "BrainGridConstants.h"


using namespace std;
using namespace mass;

BrainGrid::BrainGrid() {

}

BrainGrid::~BrainGrid() {

}


void BrainGrid::runMassSim(int size, int max_time, int interval) {
    Logger::debug("Starting MASS CUDA BrainGrid simulation\n");
    
    int nDims = 2;
	int placesSize[] = { size, size };
	Logger::debug("placesSize[0] = %d, placesSize[1] = %d", placesSize[0], placesSize[1]);
    
    // start the MASS CUDA library processes
	mass::init();

	// initialize places
	Places *neurons = mass::createPlaces<NeuronPlace, NeuronPlaceState>(0 /*handle*/, NULL /*arguments*/,
			sizeof(double), nDims, placesSize);

    neurons->callAll(NeuronPlace::SET_TIME, max_time, sizeof(int));

    // Initializes neurons as empty space or conbtaining a neuron and the type of neuron - excitatory, inhibitory, or neutral
    int* randos = neurons->calculateRandomNumbers(size*size);
    bool* emptyLocations[size*size] = { 0 };
    int numNeurons = size*size;
    
    for (int i = 0; i < size*size; ++i) {
        if (randos[i] % 100 >= BrainGridConstants::SPACE) {
            emptyLocations[i] = true;
            --numNeurons;
        }
    }

    // delete old randos 
    delete[] randos;
    randos = NULL;
    randos = neurons->calculateRandomNumbers(size * size);

    // Set neurons - SOMA or EMPTY
    neurons->callAll(NeuronPlace::INIT_NEURONS, emptyLocations, sizeof(bool) * size * size);
    // set type of neuron signal
    neurons->callAll(NeuronPlace::SET_NEURON_SIGNAL_TYPE, randos, sizeof(int) * size * size);
    delete[] randos;

    // Sets spawn times for axon and all dendrites in each neuron with SOMA
    randos = neurons->calculateRandomNumbers(size * size * MAX_NEIGHBORS);
    neurons->callAll(NeuronPlace::SET_SPAWN_TIMES, randos, sizeof(int) * size * size * MAX_NEIGHBORS)
    delete[] randos;

    // Locations for Agent instantiation 
    int* locations[numNeurons];
    for (int i = 0; i < numNeurons; ++i) {
        if (emptyLocations[i] == false) {
            locations[i] = i;
        }
    }
    
    // One Agent for each AXON and DENDRITE for each Place with SOMA
	Agents *axons = mass::createAgents<GrowingEnd, GrowingEndState> (1 /*handle*/, locations /*arguments*/,
        sizeof(int) * numNeurons, numNeurons, 0 /*placesHandle*/);
    
    Agents *dendrites = mass::createAgents<GrowingEnd, GrowingEndState> (2 /*handle*/, locations /*arguments*/, 
        sizeof(int) * numNeurons, numNeurons, 0 /*placesHandle*/);
    
    // initialize agents
    axons->callAll(GrowingEnd::INIT_AXONS);
    dendrites->callAll(GrowingEnd::INIT_DENDRITES);

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

    neurons->exchangeAll(&neighbors, NeuronPlace::FIND_AXON_GROWTH_DESTINATIONS_FROM_SOMA);

    // start a timer
	Timer timer;
    timer.start();
    
    int t = 0;
    for (; t < max_time; t++) {
        // If time to spawn, spawn
        axons->callAll(GrowingEnd::SPAWN_AXONS);
        dendrites->callAll(GrowingEnd::SPAWN_DENDRITES);
        // Is this necessary?
        dendrites->manageAll();

        // Prepare for dendrite growth
        neurons->callAll(NeuronPlace::FIND_DENDRITE_GROWTH_DESTINATIONS_FROM_SOMA);

        // Prepare for growth of Axons and Dendrites on SOMA (migrate)
        axons->callAll(GrowingEnd::GROW_AXON_SOMA);
        dendrites->callAll(GrowingEnd::GROW_DENDRITE_SOMA);

        // Grow Axons, Synapses, and Dendrites from SOMA
        axons->manageAll();
        dendrites->manageAll();

        // Set NeuronPlace traveling Dendrite or Synapse pointers that just grew from SOMA
        axons->callAll(GrowingEnd::SET_NEWLY_GROWN_SYNAPSE);
        dendrties->callAll(GrowingEnd::SET_NEWLY_GROWN_DENDRITES);
        
        randos = calculateRandomNumbers(numNeurons);
        axons->callAll(GrowingEnd::GROW_AXONS_OUTSIDE_SOMA, randos, sizeof(int) * numNeurons);
        delete[] randos;
        
        // Check if Axons switch to synapses
        randos = calculateRandomNumbers(numNeurons);
        axons->callAll(GrowingEnd::AXON_TO_SYNAPSE, randos, sizeof(int) * numNeurons);
        delete[] randos;

        randos = calculateRandomNumbers(numNeurons);
        axons->callAll(GrowingEnd::GROW_AXONS_OUTSIDE_SOMA, randos, numNeurons * sizeof(int));
        delete[] randos;

        // Check if any Dendrite-Synapse's are paired on empty neurons and growing
        //    set as pair... and not growing? Do we still need to branch (YES!)
        // neurons->callAll(NeuronPlace::MAKE_CONNECTIONS);
        
        // Try to branch into neighbors - first synapses, then dendrites
        //    each successeful branch results in a call to spawn()
        // TODO: Helper functions for numSynapses and numDendrites prior to spawning
        // TODO: Should these Agent callAll's not be Place call alls?
        randos = calculateRandomNumbers(2 * numSynapses);
        axons->callAll(GrowingEnd::BRANCH_SYNAPSES, randos, numSynapses * 2 * sizeof(int));
        delete[] randos;
        randos = calculateRandomNumbers(2 * numDendrites);
        dendrties->callAll(GrowingEnd::BRANCH_DENDRITES, randos, numDendrites * 2 * sizeof(int));
        delete[] randos;

        // TODO: Can we avoid spawning new Agent(s) when Dendrite and Synapse make a connection?

        // Grow Axons, Synapses, and Dendrites not on SOMA's
        axons->callAll(GrowingEnd::GROW_SYNAPSE);
        dendrties->CallAll(GrowingEnd::GROW_DENDRITE);

        axons->manageAll();
        dendrites->manageAll();      

        axons->callAll(GrowingEnd::SET_BRANCHED_SYNAPSES);
        dendrites->callAll(GrowingEnd::SET_BRANCHED_DENDRITES);  

        neurons->callAll(NeuronPlace::SET_BRANCH_MIGRATION_DESTINATIONS);

        axons->callAll(GrowingEnds::GROW_BRANCHES);
        dendrites->callAll(GrowingEnds::GROW_BRANCHES);

        axons->manageAll();
        dendrties->manageAll();

        // TODO: Set migrated branches
        axons->callAll(GrowingEnd::SET_MIGRATED_BRANCHES);
        dendrites3->callAll(GrowingEnd::SET_MIGRATED_BRANCHES);

        // If a synapse and dendrite meet, link them
        // TODO: Update to get SOMA Place from Dendrite Agent and kill it 
        neurons->callAll(NeuronPlace::MAKE_CONNECTIONS);

        // 1. Axon Agent travels to its SOMA
        axons->callAll(GrowingEnd::SOMA_TRAVEL);
        axons->manageAll();

        // Collect and transmit signals
        // 2. SOMAs create signal
        randos = calculateRandomNumbers(numNeurons);
        neurons->callAll(NeuronPlace::CREATE_SIGNAL, randos, numNeurons * sizeof(int));
        delete[] randos;

        // 3. axons on SOMA's get signal
        axons->callAll(GrowingEnd::GET_SIGNAL);

        // 4. synapses migrate to SOMA
        axons->callAll(GrowingEnd::DENDRITE_SOMA_TRAVEL); 
        axons->manageAll();

        // 5. dendrites place signal at SOMA
        axons->callAll(GrowingEnd::SET_SOMA_SIGNAL);

        // 6. SOMAs process received signals
        neurons->callAll(NeuronPlace::PROCESS_SIGNALS);

        // 7. Axons migrate to SOMA home
        axons->callAll(GrowingEnd::SOMA_TRAVEL);
        axons->manageAll();

        // Update end of iter params
        axons->callAll(GrowingEnd::UPDATE_ITER);
        dendrites->callAll(GrowingEnd::UPDATE_ITER);

    }

    // terminate the processes
	Mass::finish();
}