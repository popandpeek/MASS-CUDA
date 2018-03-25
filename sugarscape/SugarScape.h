
#ifndef SUGARSCAPE_H_
#define SUGARSCAPE_H_

#include "../src/Places.h"
#include "../src/Agents.h"

class SugarScape {

public:
	SugarScape();
	virtual ~SugarScape();

	void runMassSim(int size, int max_time, int interval);
	// void runDeviceSim(int size, int max_time, int interval);
	// void runHostSim(int size, int max_time, int interval);
	void displaySugar(mass::Places *places, int time, int *placesSize);
    void displayAgents(mass::Agents *agents, int time);

};

#endif /* SUGARSCAPE_H_ */
