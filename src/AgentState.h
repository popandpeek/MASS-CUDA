
#ifndef AGENTSTATE_H_
#define AGENTSTATE_H_

namespace mass {

class AgentState {
    friend class Agent;

public:
    // Place *neighbors[MAX_NEIGHBORS];  // my neighbors

    unsigned index;            // the row-major index of this place
    int size;   // the size of the Agent array

    // int message_size;  // the number of bytes in a message
    // void *inMessages[MAX_NEIGHBORS]; // holds a pointer to each neighbor's outmessage.
};

} /* namespace mass */

#endif /* AGENTSTATE_H_ */
