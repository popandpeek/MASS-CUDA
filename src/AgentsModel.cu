#include "AgentsModel.h"

namespace mass {

AgentsModel::AgentsModel(int handle, int qty) {
    Logger::debug("Running AgentsModel constructor");
    this->handle = handle;
    this->numElements = qty;
}

AgentsModel::~AgentsModel() {
    for (int i = 0; i < numElements; ++i) {
        delete agents[i];
        free(state[i]);
    }
}

std::vector<Agent**> AgentsModel::getAgentElements() {
    return agents;
}

void* AgentsModel::getStatePtr(int device) {
    Logger::debug("PlacesModel::getStatePtr: state vector size = %d", state.size());
    return state.at(device);
}

void AgentsModel::setStatePtr(std::vector<void*> src) {
    if (state.size() != 0) {
		for (int i = 0; i < nAgentsDev[i]; ++i) {
			free(state[i]);
		}
	}

	state = src;
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

