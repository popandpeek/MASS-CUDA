 #include "AgentsModel.h"

namespace mass {

AgentsModel::AgentsModel(int handle, int qty) {
    Logger::debug("Running PlacesModel constructor");
    this->handle = handle;
    this->numElements = qty;
}

AgentsModel::~AgentsModel() {
    for (int i = 0; i < numElements; ++i) {
        delete agents[i];
    }
    delete[] agents;
    free(state);
}

Agent** AgentsModel::getAgentElements() {
    return agents;
}

void* AgentsModel::getStatePtr() {
    return state;
}

int AgentsModel::getStateSize() {
    return stateBytes;
}

int AgentsModel::getHandle() {
    return handle;
}

unsigned AgentsModel::getNumElements() {
    return numElements;
}

} // end namespace

