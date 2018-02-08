#ifndef SUGARPLACESTATE_H_
#define SUGARPLACESTATE_H_

#include "../src/PlaceState.h"

class SugarPlace; //forward declaration

class SugarPlaceState: public mass::PlaceState {
public:

    int curSugar, maxSugar, nAgentsInPlace, nextAgentIdx;
    double pollution, avePollution;

    SugarPlace* migrationDest;  //available migration destination within visibility range from this place
    int migrationDestRelativeIdx;
};

#endif /* SUGARPLACESTATE_H_ */
