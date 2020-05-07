#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{

    //GraphHost g = GraphHost::GraphHost("soc-LiveJournal.txt", false);
    GraphHost g = GraphHost::GraphHost("graph-power-law-huge-2.edge", false);


    cudaProfilerStart();
    auto C = ModularityAlgorithms::Louvain(g, SORT); 
    cudaProfilerStop();
    std::cout<< "N of community found:" << C.n_of_best_communities << std::endl;
    return 0;
}
