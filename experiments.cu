

__global__ void setNeighborPlacesKernel(Place **ptrs, int nptrs, int functionId,
        void *argPtr) {
    int idx = getGlobalIdx_1D_1D();

    if (idx < nptrs) {
        PlaceState *state = ptrs[idx]->getState();
        int nSkipped = 0;
        
        #pragma unroll
        for (int i = 0; i < nNeighbors_device; ++i) {
            int j = idx + offsets_device[i];
            if (j >= 0 && j < nptrs) {
                state->neighbors[i - nSkipped] = ptrs[j];
                state->inMessages[i - nSkipped] = ptrs[j]->getMessage();
            } else {
                nSkipped++;
            }
        }


        ptrs[idx]->callMethod(functionId, argPtr);
    }
}

MASS_FUNCTION void Metal::eulerMethod() { // EULER_METHOD
    int p = myState->p;
    if (!isBorderCell()) {
        int p2 = (p + 1) % 2;

        double north = *((double*) myState->inMessages[0]);
        double east = *((double*) myState->inMessages[1]);
        double south = *((double*) myState->inMessages[2]);
        double west = *((double*) myState->inMessages[3]);

        double curTemp = myState->temp[p];
        myState->temp[p2] = curTemp + myState->r * (east - 2 * curTemp + west)
                + myState->r * (south - 2 * curTemp + north);
        //printf("eulerMethod() kernel, thread is not a border cell, neighbors: north=%d, east=%d, south=%d, west=%d. Curent temp = %d, New temp = %d\n", north, east, south, west, curTemp, myState->temp[p2]);
    } else {
        //printf("eulerMethod() kernel, thread IS a border cell. Curent temp = %d\n", myState->temp[p]);
        setBorders (p);
    }

    nextPhase();
} // end eulerMethod