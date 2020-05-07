#ifndef CONSTANTS_H     
#define CONSTANTS_H

#include <vector>

#define PRINT_DEBUG_LOG 0
#define PRINT_PERFORMANCE_LOG 1
#define CSV_FORM 1

const enum MODE { HASH, SORT, ADAPTIVE_SPEED, ADAPTIVE_MEMORY };
const int BLOCK_SIZE = 128;
const double MODULARITY_CONVERGED_THRESHOLD = 0.01;
const int EARLY_STOP_LIMIT = 1000000000;

const int STEP_ROUND = 50000000;
const unsigned long long FLAG = 18446744073709551615;
const unsigned int BUCKETS_SIZE = STEP_ROUND + STEP_ROUND / 3;



#endif 
