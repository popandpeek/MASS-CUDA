
// Maximum number of agents allowed in one place:
#define MAX_AGENTS 1

// Maximum number of neighbors each place gets data from:
#define MAX_NEIGHBORS 8

// The maximum dimentionality of the system. 
// E.g. for the 2D system MAX_DIMS should be >= 2
#define MAX_DIMS 2

// Maximum number of migration destinations for an agent from one place.
// E.g. for the system where agent can only migrate 1 cell North, South, East or West 
// N_DESTINATIONS will be 4.
#define N_DESTINATIONS 4