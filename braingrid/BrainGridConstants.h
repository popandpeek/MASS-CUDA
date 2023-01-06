#ifndef BRAINGRID_CONSTANTS_H
#define BRAINGRID_CONSTANTS_H


namespace BrainGridConstants {

    // Neuron initalization constants
    static const int EXCITATORY = 10;
    static const int INHIBITORY = 10;
    static const int NEUTRAL = 10;
    static const int SPACE = (100 - EXCITATORY - INHIBITORY - NEUTRAL);
    static const int MAX_DENDRITES = 7;

    // Neuron Signal Constants
    static const double ACTIVATING_SIGNAL = .1;
    static const double SIGNAL_MODULATION = .1;
    static const double THRESHOLD_SIG_VALUE = .4;
    
    // Neuron growth constants
    static const int AXON_GROWING_MODE = 90;
    static const int R_REPETITIVE_BRANCHES = 3;
    static const int BRANCH_POSSIBILITY = 33;
    static const int K_BRANCH_GROWTH = 5;
    static const int STOP_BRANCH_GROWTH = 10;

    // Parts of the Neuron
    // typedef enum { EMPTY = 0, SOMA, AXON, DENDRITE, SYNAPSE }NPartType;
    static const int EMPTY = 0;
    static const int SOMA = 1;
    static const int AXON = 2;
    static const int DENDRITE = 3;
    static const int SYNAPSE = 4;
    
    // typedef enum { NORTH = 0, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST }Direction;
    static const int NORTH = 0;
    static const int NORTHEAST = 1;
    static const int EAST = 2;
    static const int SOUTHEAST = 3;
    static const int SOUTH = 4;
    static const int SOUTHWEST = 5;
    static const int WEST = 6;
    static const int NORTHWEST = 7;

    //typedef enum { E = 0, I, N }SignalType;
    static const int E = 0;
    static const int I = 1;
    static const int N = 2;
}

#endif