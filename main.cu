#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{
    std::vector<std::string> dataset = { "soc-pokec-relationships.txt", "soc-LiveJournal.txt", "hollywood-2009.mtx","out.wikipedia_link_it" ,"out.wikipedia_link_fr","com-orkut.ungraph.txt", "out.dbpedia-link" };
    std::vector<int> skip = {0, 0, 49, 1, 1, 0, 2};
   
    //GraphHost g = GraphHost::GraphHost("graph-power-law-huge-2.edge", false, 0);

    for(int i = 0; i< dataset.size(); i++)
    {
        std::cout << std::endl << "****************************** NEW GRAPH ******************************" << std::endl;
        try {
            GraphHost g = GraphHost::GraphHost("dataset/" + dataset[i], false, skip[i]);
            std::cout << std::endl;

            try {
                for (int i = 0; i < 10; i++) {
                    std::cout << std::endl << "################### HASH " << i << " ##################" << std::endl;
                    auto C = ModularityAlgorithms::Louvain(g, HASH);
                    std::cout << "N of community found:" << C.n_of_best_communities << std::endl;
                }
            }
            catch (std::bad_alloc e) {
                std::cout << "Bad Alloc" << std::endl;
            }
            try {
                for (int i = 0; i < 10; i++) {
                    std::cout << std::endl << "################### SORT "<< i << " ##################" << std::endl;
                    auto C = ModularityAlgorithms::Louvain(g, SORT);
                    std::cout << "N of community found:" << C.n_of_best_communities << std::endl;
                }

            }
            catch (std::bad_alloc e) {
                std::cout << "Bad Alloc" << std::endl;
            }

        }
        catch (std::bad_alloc e) {
            std::cout << "Bad Alloc Graph" << std::endl;
            continue;
        }
      
    }


  
    return 0;
}
