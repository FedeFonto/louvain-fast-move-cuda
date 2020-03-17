#ifndef CONSTANTS_H     
#define CONSTANTS_H

#include <vector>

#define PRINT_DEBUG_LOG 1
#define PRINT_PERFORMANCE_LOG 1

const int BLOCK_SIZE = 128;
const int WARP_SIZE = 32;
const double MODULARITY_CONVERGED_THRESHOLD = 0.0000000005;
const int EARLY_STOP_LIMIT = 1000000000;
const std::vector<int> BUCKETS_RANGE = {0, 4, 8, 16, 32, 84, 319};

#endif 
