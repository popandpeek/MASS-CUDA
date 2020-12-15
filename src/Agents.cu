
#include "Agents.h"
#include "Agent.h"
#include "Dispatcher.h"
#include "Logger.h"

using namespace std;

namespace mass {

Agents::Agents(int handle, Dispatcher *d, int placesHandle) {
    this->handle = handle;
    this->dispatcher = d;

    this->elemPtrs = {};
    // BAD: No device Agents instantiated yet when this is called, causes Agent array in 
    // DeviceConfig to get set to NULL
    // this->numAgents = this->dispatcher->getNumAgentsInstantiated(handle);
    this->placesHandle = placesHandle;
}


Agents::~Agents() {
    for (int i = 0; i < elemPtrs.size(); ++i) {
        delete[] elemPtrs.at(i);
    }
}


int Agents::getNumAgents() {
    return dispatcher->getNumAgents(handle);
}

int* Agents::getNumAgentsInstantiated() {
    return dispatcher->getNumAgentsInstantiated(handle);
}

int Agents::getNumAgentObjects() {
    int totalAgents = 0;
    int* tmpAgentArr = dispatcher->getNAgentsDev(handle);
    for (int i = 0; i < (sizeof(tmpAgentArr) / sizeof(tmpAgentArr[0])); ++i) {
        totalAgents += tmpAgentArr[i];
    }

    numAgents = tmpAgentArr;
    return totalAgents;
}

int Agents::getHandle() {
    return handle;
}

void Agents::callAll(int functionId) {
    Logger::debug("Entering Agents::callAll(int functionId)");
    callAll(functionId, NULL, 0);
}

void Agents::callAll(int functionId, void *argument, int argSize) {
    Logger::debug(
            "Entering Agents::callAll(int functionId, void *argument, int argSize)");
    dispatcher->callAllAgents(handle, functionId, argument, argSize);
}


void Agents::manageAll() {
    // Step 1: kill all agents that need killing
    dispatcher->terminateAgents(handle);

    // Step 2: migrate all agents that need migrating
    dispatcher->migrateAgents(handle, placesHandle);

    // Step 3: spawn all new agents that need spawning
    dispatcher->spawnAgents(handle);
}

Agent** Agents::getElements() {
    std::vector<Agent**> elemPtrsVec = dispatcher->refreshAgents(handle); 
    mass::Agent** retVals = new Agent*[dispatcher->getNumAgents(handle)];
    int* numAgentsDev = dispatcher->getNAgentsDev(handle);
    int count = 0;
    for (int i = 0; i < elemPtrsVec.size(); ++i) {
        mass::Agent** tmp_ptr = elemPtrsVec.at(i);
        for (int j = 0; j < numAgentsDev[i]; ++j) {
            retVals[count++] = tmp_ptr[j];
        }
    }

    if (elemPtrs.size() > 0) {
        for (int i = 0; i < elemPtrs.size(); ++i) {
            delete[] elemPtrs.at(i);
        }
    }

    elemPtrs = elemPtrsVec;
    return retVals;
}

} /* namespace mass */
