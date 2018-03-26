#include "TestAgent.h"

MASS_FUNCTION TestAgent::TestAgent(mass::AgentState *state, void *argument) :
        Agent(state, argument) {
    myState = (TestAgentState*) state;
}

MASS_FUNCTION TestAgent::~TestAgent() {
    // nothing to delete
}

MASS_FUNCTION void TestAgent::callMethod(int functionId, void *argument) {
    switch (functionId) {
        case SPAWN_NORTH:
            spawn_north();
            break;
        default:
            break;
    }
}

MASS_FUNCTION TestAgentState* TestAgent::getState() {
    return myState;
}

MASS_FUNCTION void TestAgent::spawn_north() {
    TestPlaceState* placeState = (TestPlaceState*) getPlace()->state;
    if (placeState->spawningDest != NULL) {
        spawn(1, placeState->spawningDest);
        placeState->spawningDest = NULL;
    }
}


