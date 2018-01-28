#include "AgentsModel.h"

namespace mass {

AgentsModel::AgentsModel(int handle, int qty) {
    Logger::debug("Running PlacesModel constructor");
    this->handle = handle;
    this->numElements = qty;
    setIdealDims();
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

dim3 AgentsModel::blockDim() {
    return dims[0];
}

dim3 AgentsModel::threadDim() {
    return dims[1];
}

void AgentsModel::setIdealDims() {
    Logger::debug("Inside AgentsModel::setIdealDims");
    int numBlocks = (numElements - 1) / THREADS_PER_BLOCK + 1;
    dim3 blockDim(numBlocks);

    int nThr = (numElements - 1) / numBlocks + 1;
    dim3 threadDim(nThr);

    dims[0] = blockDim;
    dims[1] = threadDim;
}

} // end namespace

