
#include "Agents.h"
#include "Agent.h"
#include "Dispatcher.h"
#include "Logger.h"

using namespace std;

namespace mass {

Agents::Agents(int handle, int nAgents, Dispatcher *d) {
    this->handle = handle;
    this->dispatcher = d;

    this->elemPtrs = NULL;
    this->numElements = nAgents;
}


Agents::~Agents() {
    delete[] elemPtrs;
}


int Agents::getNumAgents() {
    //TODO: maybe should do dispatcher->refreshAgents here as well
    return numElements;
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

// void *Agents::callAll(int functionId, void *arguments[], int argSize,
//         int retSize) {
//     return dispatcher->callAllAgents(handle, functionId, arguments, argSize,
//             retSize);
// }

void Agents::manageAll() {
    //TODO: implement
    // Step 1: kill all agents that need killing
    dispatcher->terminateAgents(handle);

    // Step 2: migrate all agents that need migrating
    // Step 3: spawn all new agents that need spawning
}


Agent** Agents::getElements() {
    elemPtrs = dispatcher->refreshAgents(handle, numElements/*gets updated*/);
    return elemPtrs;
}

} /* namespace mass */
