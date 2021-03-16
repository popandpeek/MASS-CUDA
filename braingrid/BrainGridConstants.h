#ifndef BRAINGRID_CONSTANTS_H
#define BRAINGRID_CONSTANTS_H


namespace BrainGridConstants {

    // Neuron initalization constants
    const int EXCITATORY = 10;
    const int INHIBITORY = 10;
    const int NEUTRAL = 10;
    const int SPACE = (100 - EXICTATORY - INHIBITORY - NEUTRAL);
    const int MAX_DENDRITES = 7;

    // Neuron Signal Constants
    const double ACTIVATING_SIGNAL = .1;
    const double SIGNAL_MODULATION = .1;
    const double THRESHOLD_SIG_VALUE = .4;
    
    // Neuron growth constants
    const int AXON_GROWING_MODE = 90;
    const int R_REPETITIVE_BRANCHES = 3;
    const int BRANCH_POSSIBILITY = 33;
    const int K_BRANCH_GROWTH = 5;
    const int STOP_BRANCH_GROWTH = 10;

    // Parts of the Neuron
    enum NPartType { EMPTY = 0, SOMA, AXON, DENDRITE, SYNAPSE };

    enum Direction { NORTH = 0, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST, NOT_APPLICABLE };

}

#endif