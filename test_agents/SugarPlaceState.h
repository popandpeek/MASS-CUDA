#ifndef SUGARPLACESTATE_H_
#define SUGARPLACESTATE_H_

#include "../src/PlaceState.h"

class SugarPlaceState: public mass::PlaceState {
public:

    int curSugar, maxSugar, nAgentsInPlace, nextAgentIdx;
    double pollution, avePollution;
};

#endif /* SUGARPLACESTATE_H_ */
