#include "Ant.h"

MASS_FUNCTION Ant::Ant(mass::AgentState *state, void *argument) :
        Agent(state, argument) {
    myState = (AntState*) state;

    myState -> agentSugar = 0;
    myState -> agentMetabolism = 0;
    myState -> destinationIdx = 0;
}

MASS_FUNCTION Ant::~Ant() {
    // nothing to delete
}

MASS_FUNCTION void Ant::callMethod(int functionId, void *argument) {
    switch (functionId) {
        // case SET_SUGAR:
        //     setSugar();
        //     break;
        // case INC_SUGAR_AND_POLLUTION:
        //     incSugarAndPollution();
        //     break;
        // case AVE_POLLUTIONS:
        //     avePollutions();
        //     break;
        // case UPDATE_POLLUTION_WITH_AVERAGE:
        //     updatePollutionWithAverage();
        //     break;
        default:
            break;
    }
}

MASS_FUNCTION AntState* Ant::getState() {
    return myState;
}

// MASS_FUNCTION SugarPlace* Ant::getPlace() {
//     return myState->place;
// }


// MASS_FUNCTION void Ant::setPlace(SugarPlace* place) {
//     myState->place = place;
// }


