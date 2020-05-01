#ifndef CONSTANTS_H     
#define CONSTANTS_H

#include <vector>

#define PRINT_DEBUG_LOG 0
#define PRINT_PERFORMANCE_LOG 1
#define CSV_FORM 1

const int BLOCK_SIZE = 128;
const int WARP_SIZE = 32;
const double MODULARITY_CONVERGED_THRESHOLD = 0.01;
const int EARLY_STOP_LIMIT = 1000000000;
const int N_STREAM = 1;
const int STEP_ROUND = 50000000;


#endif 
