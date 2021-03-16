#ifndef BRAINGRID_H
#define BRAINGRID_H


#include "../src/Places.h"
#include "../src/Agents.h"

class BrainGrid {
    public:
        BrainGrid();
        virtual ~BrainGrid();

        void runMassSim(int size, int max_time, int interval);

}

#endif 