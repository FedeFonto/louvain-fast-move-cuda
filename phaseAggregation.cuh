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



struct AggregationPhase {
	static void run(Community& community) {
		#if PRINT_PERFORMANCE_LOG
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
		#endif


		GraphDevice new_graph = GraphDevice();

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

		new_graph.n_nodes = community_map[community_map.size() - 1];

		auto edge_source = thrust::device_vector<unsigned int>(n_edge);
		auto edge_destination = thrust::device_vector<unsigned int>(n_edge);
		auto weights = thrust::device_vector<float>(n_edge);

		auto old_start = thrust::make_zip_iterator(thrust::make_tuple(community.graph.edge_source.begin(), community.graph.edge_destination.begin(), community.graph.weights.begin()));
		auto old_end = thrust::make_zip_iterator(thrust::make_tuple(community.graph.edge_source.end(), community.graph.edge_destination.end(), community.graph.weights.end()));
		auto new_start = thrust::make_zip_iterator(thrust::make_tuple(edge_source.begin(),edge_destination.begin(), weights.begin()));

		thrust::transform(
			old_start,
			old_end,
			new_start,
			MakeCommunityPair(thrust::raw_pointer_cast(community.communities.data()), thrust::raw_pointer_cast(community_map.data()))
		);

		auto pair_iterator = thrust::make_zip_iterator(thrust::make_tuple(edge_source.begin(), edge_destination.begin()));

		thrust::sort_by_key(
			pair_iterator,
			pair_iterator + n_edge,
			weights.begin()
		);


		new_graph.edge_source = thrust::device_vector<int>(n_edge);
		new_graph.edge_destination = thrust::device_vector<int>(n_edge);
		new_graph.weights = thrust::device_vector<float>(n_edge);

		auto reduced_pair_iterator = thrust::make_zip_iterator(thrust::make_tuple(new_graph.edge_source.begin(), new_graph.edge_destination.begin()));

		auto new_end = thrust::reduce_by_key(
			pair_iterator,
			pair_iterator + n_edge,
			weights.begin(),
			reduced_pair_iterator,
			new_graph.weights.begin()
		);

		auto n_reduced_edges = new_end.first - reduced_pair_iterator;

		//printf("\n\nSTART");
		//for (int i = 0; i < n_reduced_edges; i++) {
		//	int a = thrust::get<0>((thrust::tuple<int, int>) reduced_pair_iterator[i]);
		//	int b = thrust::get<1>((thrust::tuple<int, int>) reduced_pair_iterator[i]);
		//	float c = new_graph.weights[i];
		//	printf("%d %d %.10f\n", a, b, c);
		//}

		new_graph.edge_source.resize(n_reduced_edges);
		new_graph.edge_destination.resize(n_reduced_edges);
		new_graph.weights.resize(n_reduced_edges);
		new_graph.n_links = n_reduced_edges / 2;


		new_graph.tot_weight_per_nodes = thrust::device_vector<float>(new_graph.n_nodes);

		thrust::reduce_by_key(
			new_graph.edge_source.begin(),
			new_graph.edge_source.end(),
			new_graph.weights.begin(),
			thrust::make_discard_iterator(),
			new_graph.tot_weight_per_nodes.begin()
		);


		//todo use one iteration
		new_graph.n_of_neighboor = thrust::device_vector<float>(new_graph.n_nodes);
		thrust::reduce_by_key(
			new_graph.edge_source.begin(),
			new_graph.edge_source.end(),
			thrust::make_constant_iterator(1),
			thrust::make_discard_iterator(),
			new_graph.n_of_neighboor.begin()
		);

		new_graph.total_weight = community.graph.total_weight;

		community.update(&new_graph);

#if PRINT_DEBUG_LOG
		printf("Aggregation completed\n\n");
#endif

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "Aggregation Time : " << milliseconds << "ms" << std::endl;
		#endif
	}
};

#endif
