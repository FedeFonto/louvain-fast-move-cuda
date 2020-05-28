#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{
    std::vector<std::string> dataset = { "out.patentcite",  "soc-pokec-relationships.txt", "coPapersDBLP.mtx", "packing-500x100x100-b050.mtx","soc-LiveJournal.txt", "ljournal-2008.mtx","out.wikipedia_link_ja",  "hollywood-2009.mtx","out.wikipedia_link_it" , "out.wikipedia_link_fr","com-orkut.ungraph.txt", "out.dbpedia-link", "indochina-2004.mtx" };
    std::vector<int> skip = {2, 0, 129, 141, 0, 56, 1, 49,  1, 1, 0, 2, 48 };

    for(int i = 0; i< dataset.size(); i++)
    {
        std::cout << std::endl << "****************************** NEW GRAPH ******************************" << std::endl;
        try {
            GraphHost g = GraphHost::GraphHost("dataset/" + dataset[i], false, skip[i]);
            std::cout << std::endl;
            Community C = Community(g);

            auto histo_1 = thrust::device_vector <unsigned int>(C.graph.n_of_neighboor.size(), 0);
            auto histo_2 = thrust::device_vector  <int>(C.graph.n_of_neighboor.size(), 0);

            thrust::sort(C.graph.n_of_neighboor.begin(), C.graph.n_of_neighboor.end());
            thrust::reduce_by_key(C.graph.n_of_neighboor.begin(), C.graph.n_of_neighboor.end(), thrust::constant_iterator< int>(1), histo_1.begin(), histo_2.begin());
            auto zip = thrust::make_zip_iterator(thrust::make_tuple(histo_1.begin(), histo_2.begin()));
            auto iii = thrust::make_tuple((unsigned int)0, (int)0);
            auto k = thrust::reduce(zip, zip + C.graph.n_of_neighboor.size(), iii, GetMaxValueint());
            std::cout << thrust::get<0>(k) << " " << thrust::get<1>(k) << std::endl;

        }
        catch (std::bad_alloc e) {
            std::cout << "Bad Alloc Graph" << std::endl;
            continue;
        }
      
    }


  
    return 0;
}
