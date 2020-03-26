#ifndef CONSTANTS_H     
#define CONSTANTS_H

#include <vector>

#define PRINT_DEBUG_LOG 0
#define PRINT_PERFORMANCE_LOG 0

const int BLOCK_SIZE = 128;
const int WARP_SIZE = 32;
const double MODULARITY_CONVERGED_THRESHOLD = 0.00001;
const int EARLY_STOP_LIMIT = 1000000000;

#endif 
