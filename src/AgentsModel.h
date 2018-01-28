#ifndef AGENTSMODEL_H_
#define AGENTSMODEL_H_

#define THREADS_PER_BLOCK 512

#include "Agent.h"
#include "AgentState.h"
#include "Logger.h"

namespace mass {

class AgentsModel {

public:
    virtual ~AgentsModel();

    Agent** getAgentElements();

    void* getStatePtr();
    int getStateSize();
    unsigned getNumElements();
    int getHandle();

    /**
     * Returns the ideal block dimension for this AgentsModel. Used for launching
     * kernel functions on this AgentsModel's data.
     *
     * @return
     */
    dim3 blockDim();

    /**
     * Returns the ideal thread dimension for this AgentsModel. Used for launching
     * kernel functions on this AgentsModel's data.
     *
     * @return
     */
    dim3 threadDim();

    template<typename AgentType, typename AgentStateType>
    static AgentsModel* createAgents(int handle, void *argument, 
        int argSize, int nAgents);

private:
    // initialized in createAgents function
    Agent** agents;
    void* state;
    int stateBytes;

    int handle;
    unsigned numElements;

    /*
     * Dimentions of blocks and threads for GPU
     * 0 is blockdim, 1 is threaddim
     */
    dim3 dims[2];
    
    AgentsModel(int handle, int qty);

    /**
     * Refreshes the ideal dimensions for kernel launches. This should be called
     * only when the AgentsModel is created.
     */
    void setIdealDims();
};

template<typename AgentType, typename AgentStateType>
AgentsModel* AgentsModel::createAgents(int handle, void *argument, int argSize, int nAgents) {
    Logger::debug("Entering AgentsModel::createAgents");

    AgentsModel *am = new AgentsModel(handle, nAgents);
    AgentStateType* tmpPtr = new AgentStateType[nAgents];
    am->state = tmpPtr;
    am->stateBytes = sizeof(AgentStateType);

    am->agents = new Agent*[nAgents];
    for (int i = 0; i < nAgents; ++i) {
        Agent *ag = new AgentType((AgentState*) &(tmpPtr[i]), argument);
        am->agents[i] = ag;
    }
    Logger::debug("Finished AgentsModel::createAgents");
    return am;
}

} // end namespace

#endif /* AGENTSMODEL_H_ */
