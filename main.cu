#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{

    GraphHost g = GraphHost::GraphHost("soc-LiveJournal.txt", false, 0);
    //GraphHost g = GraphHost::GraphHost("graph-power-law-huge-2.edge", false, 0);

    cudaProfilerStart();
    auto C = ModularityAlgorithms::Louvain(g, HASH); 
    std::cout << "N of community found:" << C.n_of_best_communities << std::endl <<"#####################################" << std::endl;
    C = ModularityAlgorithms::Louvain(g, SORT);
    std::cout << "N of community found:" << C.n_of_best_communities << std::endl <<"#####################################" << std::endl;
    C = ModularityAlgorithms::Louvain(g, ADAPTIVE_SPEED);
    std::cout << "N of community found:" << C.n_of_best_communities << std::endl <<"#####################################" << std::endl;
    C = ModularityAlgorithms::Louvain(g, ADAPTIVE_MEMORY);
    cudaProfilerStop();
    std::cout<< "N of community found:" << C.n_of_best_communities << std::endl;
    return 0;
}
