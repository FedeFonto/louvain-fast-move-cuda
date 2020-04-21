#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{

    GraphHost g = GraphHost::GraphHost("soc-LiveJournal.txt", false);

    cudaProfilerStart();
    auto C = ModularityAlgorithms::Laiden(g); 
    cudaProfilerStop();
    return 0;
}
