#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{
    std::vector<std::string> dataset = { "soc-LiveJournal.txt", "hollywood-2009.mtx", "europe_osm.mtx","com-orkut.ungraph.txt","uk - 2002.mtx" };  
    std::vector<int> skip = { 0, 49, 148, 0, 48 };
   
    //GraphHost g = GraphHost::GraphHost("graph-power-law-huge-2.edge", false, 0);

    for(int i = 0; i< dataset.size(); i++)
    {
        GraphHost g = GraphHost::GraphHost("soc-LiveJournal.txt", false, 0);
        std::cout << std::endl;

        auto C = ModularityAlgorithms::Louvain(g, HASH);
        std::cout << "N of community found:" << C.n_of_best_communities << std::endl << "#####################################" << std::endl;
        C = ModularityAlgorithms::Louvain(g, SORT);
        std::cout << "N of community found:" << C.n_of_best_communities << std::endl << "#####################################" << std::endl;
        C = ModularityAlgorithms::Louvain(g, ADAPTIVE_SPEED);
        std::cout << "N of community found:" << C.n_of_best_communities << std::endl << "#####################################" << std::endl;
        C = ModularityAlgorithms::Louvain(g, ADAPTIVE_MEMORY);
        std::cout << "N of community found:" << C.n_of_best_communities << std::endl;
        std::cout << std::endl << "****************************** NEW GRAPH ******************************" << std::endl;

    }


  
    return 0;
}
