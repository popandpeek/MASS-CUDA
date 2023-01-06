#include "AgentsModel.h"

namespace mass {

AgentsModel::AgentsModel(int handle, int qty, int nDevices) {
    Logger::debug("Running AgentsModel constructor");
    this->handle = handle;
    this->numElements = qty;
    this->nAgentsDev = new int[nDevices];
}

AgentsModel::~AgentsModel() {
    for (int i = 0; i < numElements; ++i) {
        delete agents[i];
        free(state[i]);
        delete nAgentsDev;
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

void AgentsModel::setNAgentsDev(int* nAgents) {
    for (int i = 0; i < agents.size(); ++i) {
        this->nAgentsDev[i] = nAgents[i];
    }
}
} // end namespace

