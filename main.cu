#include "modularityAlgorithms.h"
#include "cuda_profiler_api.h"

int main()
{
    std::vector<std::string> dataset = { "coPapersDBLP.mtx", "out.patentcite", "packing-500x100x100-b050.mtx", "soc-pokec-relationships.txt", "delaunay_n23.mtx", "soc-LiveJournal.txt" ,"out.wikipedia_link_ja",  "hollywood-2009.mtx","out.wikipedia_link_it" , "out.wikipedia_link_fr","com-orkut.ungraph.txt", "out.dbpedia-link", "indochina-2004.mtx" };
    std::vector<int> skip = { 129, 2, 141, 0, 131, 0, 1, 49,  1, 1, 0, 2, 48 };

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

            auto max_val = C.graph.n_of_neighboor.back();
            auto k = thrust::reduce_by_key(C.graph.n_of_neighboor.begin(), C.graph.n_of_neighboor.end(), thrust::constant_iterator< int>(1), histo_1.begin(), histo_2.begin());
            auto len = k.first - histo_1.begin();

            auto max_pow = 2;
            while (max_pow < max_val)
                max_pow = max_pow * 2;

            auto histo_3 = thrust::device_vector  <int>(max_pow, 0);
            for (int h = 0; h < len; h++ ) {
                histo_3[histo_1[h]] = histo_2[h];
            }

            for (int h = 0; h < max_pow; h++) {
                std::cout << h << " " << histo_3[h] << std::endl;
            }
        }
        catch (std::bad_alloc e) {
            std::cout << "Bad Alloc Graph" << std::endl;
            continue;
        }
      
    }


  
    return 0;
}
