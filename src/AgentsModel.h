#ifndef AGENTSMODEL_H_
#define AGENTSMODEL_H_

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
    
    AgentsModel(int handle, int qty);
};

template<typename AgentType, typename AgentStateType>
AgentsModel* AgentsModel::createAgents(int handle, void *argument, int argSize, 
    int nAgents, std::vector<DeviceConfig> devices) {
    Logger::debug("Entering AgentsModel::createAgents");

    AgentsModel *am = new AgentsModel(handle, nAgents);

/*    std::vector<int>::iterator itr;
    for (itr = devices.begin(); itr < devices.end(); itr++) {
        cudaSetDevice(itr->getDeviceNum());
        itr->instantiateAgents<AgentType, AgentStateType>(handle, argument, argSize, 
            nAgents / devices.size(), placesHandle, placeIdxs);
    }*/

    // AgentStateType* tmpPtr = new AgentStateType[nAgents];
    // am->state = tmpPtr;
    // am->stateBytes = sizeof(AgentStateType);

    // am->agents = new Agent*[nAgents];
    // for (int i = 0; i < nAgents; ++i) {
    //     Agent *ag = new AgentType((AgentState*) &(tmpPtr[i]), argument);
    //     am->agents[i] = ag;
    // }
    // Logger::debug("Finished AgentsModel::createAgents");
    
    return am;
}

} // end namespace

#endif /* AGENTSMODEL_H_ */
