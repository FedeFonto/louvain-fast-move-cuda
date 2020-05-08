#ifndef OPTIMIZATION_SORT_CU     
#define OPTIMIZATION_SORT_CU

#include "phaseOptimization.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
static void select_self_node(
	unsigned int n_nodes,
	unsigned int* source,
	unsigned int* dest,
	unsigned int* community,
	float* value,
	float* self
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_nodes) {
		int node = source[id];
		if (community[node] == dest[id]) {
			self[node] = value[id];
		}
	}
};


void OptimizationPhase::optimize_fast() {
#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
	cudaEvent_t  round_start, copy, sort, reduce_sort, self_c, transform, reduce_transform;
	cudaEventCreate(&round_start);
	cudaEventCreate(&copy);
	cudaEventCreate(&sort);
	cudaEventCreate(&reduce_sort);
	cudaEventCreate(&self_c);
	cudaEventCreate(&transform);
	cudaEventCreate(&reduce_transform);
	float milliseconds = 0;
#endif

	int limit_round;
	int round = 0;


	while (round < community.graph.edge_destination.size()) {
#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(round_start);
#endif
		key_node_source.resize(STEP_ROUND);
		key_community_dest.resize(STEP_ROUND);
		values_weight.resize(STEP_ROUND);

		limit_round = round + STEP_ROUND;
		if (limit_round >= community.graph.edge_destination.size()) {
			limit_round = community.graph.edge_destination.size();
		}
		else {
			limit_round = community.graph.neighboorhood_sum[community.graph.edge_source[limit_round] - 1];
		}

		auto key_community = thrust::make_zip_iterator(thrust::make_tuple(key_node_source.begin(), key_community_dest.begin()));
		auto selected_edge = thrust::make_zip_iterator(thrust::make_tuple(key_node_source.begin(), key_community_dest.begin(), values_weight.begin()));

		int n_edge_in_buckets;

		auto p = thrust::copy_if(
			thrust::make_transform_iterator(community.start + round, MakeCommunityDest(thrust::raw_pointer_cast(community.communities.data()))),
			thrust::make_transform_iterator(community.start + limit_round, MakeCommunityDest(thrust::raw_pointer_cast(community.communities.data()))),
			thrust::make_zip_iterator(thrust::make_tuple(community.graph.edge_source.begin() + round, community.graph.edge_destination.begin() + round)),
			selected_edge,
			TestTupleValue(thrust::raw_pointer_cast(neighboorhood_change.data()))
		);

		n_edge_in_buckets = p - selected_edge;

		key_node_source.resize(n_edge_in_buckets);
		key_community_dest.resize(n_edge_in_buckets);
		values_weight.resize(n_edge_in_buckets);


#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(copy);
		cudaEventSynchronize(copy);
		cudaEventElapsedTime(&milliseconds, round_start, copy);
#if CSV_FORM
		std::cout << n_edge_in_buckets << "," << community.graph.edge_source.size() << "," << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << ",";
		std::cout << milliseconds << ",";
#else
		std::cout << "\nNumber of Edges selected: " << n_edge_in_buckets << " / " << community.graph.edge_source.size() << " (" << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << " %)" << std::endl;
		std::cout << " - Copy Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		if (n_edge_in_buckets == 0) {
#if PRINT_DEBUG_LOG
			printf("No elements!\n");
#endif 
			return;
		}


#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(sort);
		cudaEventSynchronize(sort);
		cudaEventElapsedTime(&milliseconds, copy, sort);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Sort Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(reduce_sort);
		cudaEventSynchronize(reduce_sort);
		cudaEventElapsedTime(&milliseconds, sort, reduce_sort);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Reduce after sort time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		auto self_community = thrust::device_vector<float>(community.graph.n_nodes, 0);


#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(self_c);
		cudaEventSynchronize(self_c);
		cudaEventElapsedTime(&milliseconds, reduce_sort, self_c);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Obtain self communities : " << milliseconds << "ms" << std::endl;
#endif
#endif

		thrust::transform(
			selected_edge,
			selected_edge + n_edge_in_buckets,
			values_weight.begin(),
			DeltaModularitySort(
				thrust::raw_pointer_cast(community.communities_weight.data()),
				thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
				community.graph.total_weight,
				thrust::raw_pointer_cast(self_community.data()),
				thrust::raw_pointer_cast(community.communities.data())
			)
		);

		/*for (int i = 0; i < n_reduced_edges; i++) {
			std::cout << reduced_key_source[i] << " " << community.communities[reduced_key_source[i]] << " " << reduced_key_dest[i] << " " << reduced_value[i] << std::endl;
		}*/

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(transform);
		cudaEventSynchronize(transform);
		cudaEventElapsedTime(&milliseconds, self_c, transform);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Transform Time : " << milliseconds << "ms" << std::endl;
#endif
#endif



		auto community_value_pair_input = thrust::make_zip_iterator(thrust::make_tuple(key_community_dest.begin(), values_weight.begin()));
		auto community_value_pair_output = thrust::make_zip_iterator(thrust::make_tuple(final_community.begin() + nodes_considered, final_value.begin() + nodes_considered));

		auto ne = thrust::reduce_by_key(
			key_node_source.begin(),
			key_node_source.begin() + n_edge_in_buckets,
			community_value_pair_input,
			final_node.begin() + nodes_considered,
			community_value_pair_output,
			thrust::equal_to<int>(),
			GetMaxValue()
		);

		nodes_considered = ne.first - final_node.begin();
		round = limit_round;

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(reduce_transform);
		cudaEventSynchronize(reduce_transform);
		cudaEventElapsedTime(&milliseconds, transform, reduce_transform);
#if CSV_FORM
		std::cout << milliseconds << "," << std::endl;
#else
		std::cout << " - Reduce after transform Time : " << milliseconds << "ms" << std::endl;
#endif
#endif
	}
}


void OptimizationPhase::optimize_sort() {
#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
	cudaEvent_t  round_start, copy, sort, reduce_sort, self_c, transform, reduce_transform;
	cudaEventCreate(&round_start);
	cudaEventCreate(&copy);
	cudaEventCreate(&sort);
	cudaEventCreate(&reduce_sort);
	cudaEventCreate(&self_c);
	cudaEventCreate(&transform);
	cudaEventCreate(&reduce_transform);
	float milliseconds = 0;
#endif

	int limit_round;
	int round = 0;


	while (round < community.graph.edge_destination.size()) {
#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(round_start);
#endif
		key_node_source.resize(STEP_ROUND);
		key_community_dest.resize(STEP_ROUND);
		values_weight.resize(STEP_ROUND);

		reduced_key_source.resize(STEP_ROUND);
		reduced_key_dest.resize(STEP_ROUND);
		reduced_value.resize(STEP_ROUND);

		limit_round = round + STEP_ROUND;
		if (limit_round >= community.graph.edge_destination.size()) {
			limit_round = community.graph.edge_destination.size();
		}
		else {
			limit_round = community.graph.neighboorhood_sum[community.graph.edge_source[limit_round] - 1];
		}

		auto key_community = thrust::make_zip_iterator(thrust::make_tuple(key_node_source.begin(), key_community_dest.begin()));
		auto selected_edge = thrust::make_zip_iterator(thrust::make_tuple(key_node_source.begin(), key_community_dest.begin(), values_weight.begin()));

		int n_edge_in_buckets;

		auto p = thrust::copy_if(
			thrust::make_transform_iterator(community.start + round, MakeCommunityDest(thrust::raw_pointer_cast(community.communities.data()))),
			thrust::make_transform_iterator(community.start + limit_round, MakeCommunityDest(thrust::raw_pointer_cast(community.communities.data()))),
			thrust::make_zip_iterator(thrust::make_tuple(community.graph.edge_source.begin() + round, community.graph.edge_destination.begin() + round)),
			selected_edge,
			TestTupleValue(thrust::raw_pointer_cast(neighboorhood_change.data()))
		);

		n_edge_in_buckets = p - selected_edge;

		key_node_source.resize(n_edge_in_buckets);
		key_community_dest.resize(n_edge_in_buckets);
		values_weight.resize(n_edge_in_buckets);


#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(copy);
		cudaEventSynchronize(copy);
		cudaEventElapsedTime(&milliseconds, round_start, copy);
#if CSV_FORM
		std::cout << n_edge_in_buckets << "," << community.graph.edge_source.size() << "," << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << ",";
		std::cout << milliseconds << ",";
#else
		std::cout << "\nNumber of Edges selected: " << n_edge_in_buckets << " / " << community.graph.edge_source.size() << " (" << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << " %)" << std::endl;
		std::cout << " - Copy Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		if (n_edge_in_buckets == 0) {
#if PRINT_DEBUG_LOG
			printf("No elements!\n");
#endif 
			return;
		}


		int step = 5000000;
		int i = 0;
		for (int off = 0; off < n_edge_in_buckets; off += step) {
			int limit = off + step;
			if (limit >= n_edge_in_buckets) {
				limit = n_edge_in_buckets;
			}
			else {
				limit += community.graph.n_of_neighboor[key_node_source[limit]];
				if (limit >= n_edge_in_buckets) {
					limit = n_edge_in_buckets;
				}
			}
			thrust::sort_by_key(
				key_community + off,
				key_community + limit,
				values_weight.begin() + off
			);
			i++;
		}



#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(sort);
		cudaEventSynchronize(sort);
		cudaEventElapsedTime(&milliseconds, copy, sort);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Sort Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		auto reduced_key = thrust::make_zip_iterator(thrust::make_tuple(reduced_key_source.begin(), reduced_key_dest.begin()));
		auto reduced_list = thrust::make_zip_iterator(thrust::make_tuple(reduced_key_source.begin(), reduced_key_dest.begin(), reduced_value.begin()));

		auto new_end = thrust::reduce_by_key(
			key_community,
			key_community + n_edge_in_buckets,
			values_weight.begin(),
			reduced_key,
			reduced_value.begin()
		);

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(reduce_sort);
		cudaEventSynchronize(reduce_sort);
		cudaEventElapsedTime(&milliseconds, sort, reduce_sort);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Reduce after sort time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		auto n_reduced_edges = new_end.first - reduced_key;
		reduced_key_source.resize(n_reduced_edges);
		reduced_key_dest.resize(n_reduced_edges);
		reduced_value.resize(n_reduced_edges);

		auto self_community = thrust::device_vector<float>(community.graph.n_nodes);

		int n_blocks = (n_reduced_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;


		select_self_node << <n_blocks, BLOCK_SIZE >> > (
			n_reduced_edges,
			thrust::raw_pointer_cast(reduced_key_source.data()),
			thrust::raw_pointer_cast(reduced_key_dest.data()),
			thrust::raw_pointer_cast(community.communities.data()),
			thrust::raw_pointer_cast(reduced_value.data()),
			thrust::raw_pointer_cast(self_community.data())
			);

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(self_c);
		cudaEventSynchronize(self_c);
		cudaEventElapsedTime(&milliseconds, reduce_sort, self_c);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Obtain self communities : " << milliseconds << "ms" << std::endl;
#endif
#endif

		thrust::transform(
			reduced_list,
			reduced_list + n_reduced_edges,
			reduced_value.begin(),
			DeltaModularitySort(
				thrust::raw_pointer_cast(community.communities_weight.data()),
				thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
				community.graph.total_weight,
				thrust::raw_pointer_cast(self_community.data()),
				thrust::raw_pointer_cast(community.communities.data())
			)
		);

		/*for (int i = 0; i < n_reduced_edges; i++) {
			std::cout << reduced_key_source[i] << " " << community.communities[reduced_key_source[i]] << " " << reduced_key_dest[i] << " " << reduced_value[i] << std::endl;
		}*/

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(transform);
		cudaEventSynchronize(transform);
		cudaEventElapsedTime(&milliseconds, self_c, transform);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else
		std::cout << " - Transform Time : " << milliseconds << "ms" << std::endl;
#endif
#endif



		auto community_value_pair_input = thrust::make_zip_iterator(thrust::make_tuple(reduced_key_dest.begin(), reduced_value.begin()));
		auto community_value_pair_output = thrust::make_zip_iterator(thrust::make_tuple(final_community.begin() + nodes_considered, final_value.begin() + nodes_considered));

		auto ne = thrust::reduce_by_key(
			reduced_key_source.begin(),
			reduced_key_source.begin() + n_reduced_edges,
			community_value_pair_input,
			final_node.begin() + nodes_considered,
			community_value_pair_output,
			thrust::equal_to<int>(),
			GetMaxValue()
		);

		nodes_considered = ne.first - final_node.begin();
		round = limit_round;

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(reduce_transform);
		cudaEventSynchronize(reduce_transform);
		cudaEventElapsedTime(&milliseconds, transform, reduce_transform);
#if CSV_FORM
		std::cout << milliseconds << "," << std::endl;
#else
		std::cout << " - Reduce after transform Time : " << milliseconds << "ms" << std::endl;
#endif
#endif
	}
}



#endif