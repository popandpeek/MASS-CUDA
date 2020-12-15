#ifndef AGENTSMODEL_H_
#define AGENTSMODEL_H_

#include "Agent.h"
#include "AgentState.h"
#include "Logger.h"
#include <vector>

namespace mass {

class AgentsModel {

public:
    virtual ~AgentsModel();

    std::vector<Agent**> getAgentElements();
    void* getStatePtr(int);
    void setStatePtr(std::vector<void*>);
    int getStateSize();
    unsigned getNumElements();
    int getHandle();

    template<typename AgentType, typename AgentStateType>
    static AgentsModel* createAgents(int handle, void *argument, 
        int argSize, int nAgents, int* nAgentsDev, int nDevices);

private:
    // initialized in createAgents function
    std::vector<Agent**> agents;
    std::vector<void*> state;
    int stateBytes;
    int handle;
    unsigned numElements;
    int* nAgentsDev; // tracking array for agents on each device
    AgentsModel(int handle, int nAgents);
};

template<typename AgentType, typename AgentStateType>
AgentsModel* AgentsModel::createAgents(int handle, void *argument, int argSize, int nAgents, int* nAgentsDev, int nDevices) {
    Logger::debug("Entering AgentsModel::createAgents");

    AgentsModel *am = new AgentsModel(handle, nAgents);
    am->nAgentsDev = nAgentsDev;
    am->stateBytes = sizeof(AgentStateType);
    Logger::debug("AgentsModel::createAgents: nAgentsDev = %d; State bytes = %d", am->nAgentsDev[0] + am->nAgentsDev[1], am->stateBytes);
    for (int i = 0; i < nDevices; ++i) {
        Agent** a_ptrs = new Agent*[am->nAgentsDev[i]];
        AgentStateType* tmpPtr = new AgentStateType[am->nAgentsDev[i]];
        for (int j = 0; j < am->nAgentsDev[i]; ++j) {
            Agent *ag = new AgentType((AgentState*) &(tmpPtr[j]), argument);
            a_ptrs[j] = ag;
        }
        
        am->agents.push_back(a_ptrs);
        am->state.push_back(tmpPtr);
    }
    Logger::debug("Finished AgentsModel::createAgents");
    return am;
}


} // end namespace

#endif /* AGENTSMODEL_H_ */