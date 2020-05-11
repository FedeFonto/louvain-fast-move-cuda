#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{
    std::vector<std::string> dataset = { "com-orkut.ungraph.txt", "out.trackers", "out.wikipedia_link_fr",  "soc-LiveJournal.txt","hollywood-2009.mtx","out.wikipedia_link_it" ,"soc-pokec-relationships.txt", "out.dbpedia-link" };  
    std::vector<int> skip = { 0, 1, 1, 0, 49,1,0,2};
   
    //GraphHost g = GraphHost::GraphHost("graph-power-law-huge-2.edge", false, 0);

    for(int i = 0; i< dataset.size(); i++)
    {
        std::cout << std::endl << "****************************** NEW GRAPH ******************************" << std::endl;
        try {
            GraphHost g = GraphHost::GraphHost("dataset/" + dataset[i], false, skip[i]);
            std::cout << std::endl;

            try {
                std::cout << std::endl << "################### HASH ##################" << std::endl;
                auto C = ModularityAlgorithms::Louvain(g, HASH);
                std::cout << "N of community found:" << C.n_of_best_communities << std::endl;
            }
            catch (std::bad_alloc e) {
                std::cout << "Bad Alloc" << std::endl;
            }
            try {
                std::cout << std::endl << "################### SORT ##################" << std::endl;
                auto C = ModularityAlgorithms::Louvain(g, SORT);
                std::cout << "N of community found:" << C.n_of_best_communities << std::endl;

            }
            catch (std::bad_alloc e) {
                std::cout << "Bad Alloc" << std::endl;
            }
            try {
                std::cout << std::endl << "################### ADAPTIVE SPEED ##################" << std::endl;
                auto C = ModularityAlgorithms::Louvain(g, ADAPTIVE_SPEED);
                std::cout << "N of community found:" << C.n_of_best_communities << std::endl;
            }
            catch (std::bad_alloc e) {
                std::cout << "Bad Alloc" << std::endl <<std::endl;
            }
            try {
                std::cout << std::endl << "################### ADAPTIVE MEMORY ##################" << std::endl;
                auto C = ModularityAlgorithms::Louvain(g, ADAPTIVE_MEMORY);
                std::cout << "N of community found:" << C.n_of_best_communities << std::endl ;

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
