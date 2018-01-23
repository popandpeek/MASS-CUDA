/*
 *  @file MetalState.h
 	
 *	@section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef SUGARPLACESTATE_H_
#define SUGARPLACESTATE_H_

#include "../src/PlaceState.h"

class SugarPlaceState: public mass::PlaceState {
public:

    int curSugar, maxSugar, nAgentsInPlace, nextAgentIdx;
    double pollution, avePollution;
    
    // Agent properties:
    // int agentSugar, agentMetabolism, destinationIdx;
};

#endif /* SUGARPLACESTATE_H_ */
