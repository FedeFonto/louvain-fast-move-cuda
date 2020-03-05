#ifndef OPTIMIZATION_CU     
#define OPTIMIZATION_CU

#include "phaseOptimization.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void update_kernel(
	int n_nodes,
	int* nodes_to_update,
	int* community_dest,
	float* delta_value,
	int* community,
	float* community_weight,
	float* nodes_weight,
	bool* n_change,
	int* edges,
	unsigned long* degrees
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_nodes) {
		int node = nodes_to_update[id];
		int c = community_dest[id];
		if (community[node] == c || delta_value <= 0) {
			return;
		}
		else {
			atomicAdd(&community_weight[c], nodes_weight[node]);
			atomicAdd(&community_weight[community[node]], nodes_weight[node] * -1);
			community[node] = c;

			for (int i = degrees[node]; i < degrees[node + 1]; node++) {
				n_change[edges[i]] = true;
			}
		}
	}
};


void OptimizationPhase::optimize() {

	for (int j = 0; j <= BUCKETS_RANGE.size(); j++) {

		//todo find a way to using less space
		auto key_community_source = thrust::device_vector<int>(community.graph.edge_source.size());
		auto key_community_dest = thrust::device_vector<int>(community.graph.edge_source.size());
		auto key_community = thrust::make_zip_iterator(thrust::make_tuple(key_community_source.begin(), key_community_dest.begin()));


		auto values_weight = thrust::device_vector<float>(community.graph.edge_source);
		auto selected_edge = thrust::make_zip_iterator(thrust::make_tuple(key_community_source.begin(), key_community_dest.begin(), values_weight.begin()));
		
		auto p = thrust::copy_if(
			thrust::make_transform_iterator(community.start, MakeCommunityDest(thrust::raw_pointer_cast(community.communities.data()))),
			thrust::make_transform_iterator(community.end, MakeCommunityDest(thrust::raw_pointer_cast(community.communities.data()))),
			community.graph.edge_source.begin(),
			selected_edge,
			TestTupleValue(thrust::raw_pointer_cast(buckets.data()), thrust::raw_pointer_cast(neighboorhood_change.data()), j)
		);


		int n_edge_in_buckets = p - selected_edge;
		key_community_source.resize(n_edge_in_buckets);
		key_community_dest.resize(n_edge_in_buckets);
		values_weight.resize(n_edge_in_buckets);

		if (n_edge_in_buckets == 0) {
			printf("No elements in bucket %d!\n", j);
			continue;
		}

		thrust::sort_by_key(
			key_community,
			key_community + n_edge_in_buckets,
			values_weight.begin()
		);

		auto reduced_key_source = thrust::device_vector<int>(n_edge_in_buckets);
		auto reduced_key_dest = thrust::device_vector<int>(n_edge_in_buckets);
		auto reduced_value = thrust::device_vector<float>(n_edge_in_buckets);

		auto reduced_key = thrust::make_zip_iterator(thrust::make_tuple(reduced_key_source.begin(), reduced_key_dest.begin()));

		auto new_end = thrust::reduce_by_key(
			key_community,
			key_community + n_edge_in_buckets,
			values_weight.begin(),
			reduced_key,
			reduced_value.begin()
		);


		auto n_reduced_edges = new_end.first - reduced_key;
		reduced_key_source.resize(n_reduced_edges);
		reduced_key_dest.resize(n_reduced_edges);
		reduced_value.resize(n_reduced_edges);

		auto reduced_list = thrust::make_zip_iterator(thrust::make_tuple(reduced_key_source.begin(), reduced_key_dest.begin(), reduced_value.begin()));

		//printf("\n\n%d ,%d \n", n_reduced_edges, n_edge_in_buckets);
		//for (int i = 0; i < n_reduced_edges; i++) {
		//	try {
		//		int a = thrust::get<0>(reduced_list[i]);
		//		int b = thrust::get<1>(reduced_list[i]);
		//		std::cout << a << " " << b << " " << community.graph.tot_weight_per_nodes[a] <<" "<< community.communities_weight[b] << std::endl;
		//	}
		//	catch (thrust::system_error e) {
		//		std::cout << "Error " << i << std::endl;
		//	}
		//}

		thrust::transform(
			reduced_list,
			reduced_list + n_reduced_edges,
			reduced_value.begin(),
			DeltaModularity(
				thrust::raw_pointer_cast(community.communities_weight.data()),
				thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
				community.graph.total_weight
			)
		);
			

		auto community_value_pair_input = thrust::make_zip_iterator(thrust::make_tuple(reduced_key_dest.begin(), reduced_value.begin()));
		auto community_value_pair_output = thrust::make_zip_iterator(thrust::make_tuple(key_community_dest.begin(), values_weight.begin()));

		auto ne = thrust::reduce_by_key(
			reduced_key_source.begin(),
			reduced_key_source.begin() + n_reduced_edges,
			community_value_pair_input,
			key_community_source.begin(),
			community_value_pair_output,
			thrust::equal_to<int>(),
			GetMaxValue()
		);


		n_reduced_edges = ne.first - key_community_source.begin();

		int n_blocks = (n_reduced_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

		update_kernel << <n_blocks, BLOCK_SIZE >> > (
			n_reduced_edges,
			thrust::raw_pointer_cast(key_community_source.data()),
			thrust::raw_pointer_cast(key_community_dest.data()),
			thrust::raw_pointer_cast(reduced_value.data()),
			thrust::raw_pointer_cast(community.communities.data()),
			thrust::raw_pointer_cast(community.communities_weight.data()),
			thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
			thrust::raw_pointer_cast(neighboorhood_change.data()),
			thrust::raw_pointer_cast(community.graph.edge_destination.data()),
			thrust::raw_pointer_cast(community.graph.degrees.data())
			);
		
	}
}



#endif