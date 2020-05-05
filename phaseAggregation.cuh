#ifndef AGGREGATION_H     
#define AGGREGATION_H

#include "community.h"
#include "constants.h"
#include "operators.h"
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>
#include <thrust/functional.h>


struct AggregationPhase {
	static void run(Community& community, bool aggregate) {
		#if PRINT_PERFORMANCE_LOG
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
		#endif

		int n_edge = community.graph.edge_source.size();
		auto community_map = thrust::device_vector<unsigned int>(community.communities.size());

		thrust::transform(
			community.communities_weight.begin(),
			community.communities_weight.end(),
			community_map.begin(),
			CountNotZero()
		);

	
		//prefix sum
		thrust::inclusive_scan(
			community_map.begin(),
			community_map.end(),
			community_map.begin()
		);

		community.update_best(thrust::raw_pointer_cast(community_map.data()));
		community.n_of_best_communities = community_map.back();

		if (aggregate) {
			community.graph.n_nodes = community_map[community_map.size() - 1];

			HashMap h = HashMap(community.graph.edge_destination.size() + community.graph.edge_destination.size() / 3);

			h.fill_for_aggregation(
				thrust::raw_pointer_cast(community.graph.edge_source.data()),
				thrust::raw_pointer_cast(community.graph.edge_destination.data()),
				thrust::raw_pointer_cast(community.graph.weights.data()),
				community.graph.edge_source.size(),
				thrust::raw_pointer_cast(community.communities.data()),
				thrust::raw_pointer_cast(community_map.data())
			);

			auto n_reduced_edges = h.resize();

			thrust::sort_by_key(
				h.key.begin(),
				h.key.begin() + n_reduced_edges,
				h.values.begin()
			);

			community.graph.edge_source = thrust::device_vector<int>(n_reduced_edges);
			community.graph.edge_destination = thrust::device_vector<int>(n_reduced_edges);
			community.graph.weights = thrust::device_vector<float>(n_reduced_edges);

			auto zip = thrust::make_zip_iterator(thrust::make_tuple(h.key.begin(), h.values.begin()));
			auto iter = thrust::make_transform_iterator(zip, SplitIterator());
			thrust::copy(iter, iter + n_reduced_edges, community.start);

			/*printf("\n\nSTART");
			for (int i = 0; i < n_reduced_edges; i++) {
				int a = community.graph.edge_source[i];
				int b = community.graph.edge_destination[i];
				float c = community.graph.weights[i];
				printf("%d %d %.10f\n", a, b, c);
			}*/

			community.graph.edge_source.resize(n_reduced_edges);
			community.graph.edge_destination.resize(n_reduced_edges);
			community.graph.weights.resize(n_reduced_edges);
			community.graph.n_links = n_reduced_edges / 2;


			community.graph.tot_weight_per_nodes = thrust::device_vector<float>(community.graph.n_nodes);

			auto value_input = thrust::make_zip_iterator(thrust::make_tuple(community.graph.weights.begin(), thrust::make_constant_iterator((unsigned int) 1)));
			auto value_output = thrust::make_zip_iterator(thrust::make_tuple(community.graph.tot_weight_per_nodes.begin(), community.graph.n_of_neighboor.begin()));

			community.graph.n_of_neighboor.resize(community.graph.n_nodes);
			community.graph.tot_weight_per_nodes.resize(community.graph.n_nodes);
			community.graph.neighboorhood_sum.resize(community.graph.n_nodes);

			thrust::reduce_by_key(
				community.graph.edge_source.begin(),
				community.graph.edge_source.end(),
				value_input,
				thrust::make_discard_iterator(),
				value_output,
				thrust::equal_to<unsigned int>(),
				PairSum<float,unsigned int>()
			);


			thrust::exclusive_scan(
				community.graph.n_of_neighboor.begin(),
				community.graph.n_of_neighboor.end(),
				community.graph.neighboorhood_sum.begin()
			);

			community.graph.total_weight = community.graph.total_weight;

			community.update();

		}

#if PRINT_DEBUG_LOG
		printf("Aggregation completed\n\n");
#endif

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "\nAggregation Time : " << milliseconds << "ms" << std::endl;
		#endif
	}
};

#endif
