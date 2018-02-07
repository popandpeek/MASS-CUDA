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
        case MIGRATE:
            migrate();
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

    SugarPlace* myPlace = (SugarPlace*) getPlace();

    myState -> agentSugar += myPlace -> getCurSugar();
    myState -> agentSugar -= myState -> agentMetabolism;

    myPlace -> setCurSugar(0);
    myPlace -> setPollution(myPlace -> getPollution() + myState->agentMetabolism);

    //TODO: experiment to see if direct access to variables is faster than accessors

    if( myState -> agentSugar < 0 )
    {
        terminateAgent();
    }
}

MASS_FUNCTION void Ant::migrate() {
    // printf("migrate() function for agent %d\n", getIndex());
    SugarPlace* myPlace = (SugarPlace*) getPlace();
    if (myPlace -> getMigrationDest() != NULL) {
        migrateAgent(myPlace -> getMigrationDest(), myPlace -> getMigrationDestRelIdx());
    }
}


