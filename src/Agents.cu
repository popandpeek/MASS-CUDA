
#include "Agents.h"
#include "Agent.h"
#include "Dispatcher.h"
#include "Logger.h"

using namespace std;

namespace mass {

Agents::Agents(int handle, Dispatcher *d, int placesHandle) {
    this->handle = handle;
    this->dispatcher = d;

    this->elemPtrs = NULL;
    //this->numElements = nAgents;
    this->placesHandle = placesHandle;
}


Agents::~Agents() {
    delete[] elemPtrs;
}


int Agents::getNumAgents() {
    return dispatcher->getNumAgents(handle);
}

int Agents::getNumAgentObjects() {
    return dispatcher->getNumAgentObjects(handle);
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
    //TODO: implement
    // Step 1: kill all agents that need killing
    dispatcher->terminateAgents(handle);

    // Step 2: migrate all agents that need migrating
    dispatcher->migrateAgents(handle, placesHandle);

    // Step 3: spawn all new agents that need spawning
    dispatcher->spawnAgents(handle);
}


Agent** Agents::getElements() {
    elemPtrs = dispatcher->refreshAgents(handle);
    return elemPtrs;
}

} /* namespace mass */
