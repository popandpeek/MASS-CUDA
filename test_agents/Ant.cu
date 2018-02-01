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
        case METABOLIZE:
            metabolize();
            break;
        case SET_INIT_SUGAR:
            setInitSugar((int*)argument);
            break;
        case SET_INIT_METABOLISM:
            setInitMetabolism((int*)argument);
            break;
        default:
            break;
    }
}

MASS_FUNCTION AntState* Ant::getState() {
    return myState;
}

MASS_FUNCTION void Ant::setInitSugar(int *agentSugarArray) {

    myState -> agentSugar = agentSugarArray[getIndex()]; 
}

MASS_FUNCTION void Ant::setInitMetabolism(int *agentMetabolismArray) {
    myState -> agentMetabolism = agentMetabolismArray[getIndex()];
}

MASS_FUNCTION void Ant::metabolize() {
    //TODO: implement

    SugarPlace* myPlace = (SugarPlace*) getPlace();

    myState -> agentSugar += myPlace -> getCurSugar();
    myState -> agentSugar -= myState -> agentMetabolism;

    myPlace -> setCurSugar(0); //TODO: implement
    myPlace -> setPollution(myPlace -> getPollution() + myState->agentMetabolism); //TODO: implememnt functions in the SugarPlace

    //TODO: experiment to see if direct access to variables is faster than accessors

    if( myState -> agentSugar < 0 )
    {
        // Kill agent

        //TODO: update with proper killing routine when available

        // nAgentsInPlace[idx] = 0;
        // myState -> agentSugar = -1;
        // myState -> agentMetabolism = -1;
    }
}


