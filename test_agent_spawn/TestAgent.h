
#include "../src/Agent.h"
#include "../src/AgentState.h"
#include "../src/Logger.h"
#include "TestPlace.h"

class TestAgentState: public mass::AgentState {};

class TestAgent: public mass::Agent {

public:

    const static int SPAWN_NORTH = 0;

    MASS_FUNCTION TestAgent(mass::AgentState *state, void *argument);
    MASS_FUNCTION ~TestAgent();

    MASS_FUNCTION virtual void callMethod(int functionId, void *arg = NULL);
    MASS_FUNCTION virtual TestAgentState* getState();

private:
    TestAgentState* myState;

    MASS_FUNCTION void spawn_north();
};

