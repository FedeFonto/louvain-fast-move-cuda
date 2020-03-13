#ifndef OPTIMIZATION_CU     
#define OPTIMIZATION_CU

#include "phaseOptimization.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void update_value_kernel(
	int n_nodes,
	int* nodes_to_update,
	int* community_dest,
	float* delta_value,
	int* community,
	float* community_weight,
	float* nodes_weight,
	bool* its_changed
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
			its_changed[node] = true;
		}
	}
};

__global__
void update_changed_kernel(
	bool* n_changed,
	bool* its_changed,
	int* source,
	int* dest,
	int n_edge
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_edge) {
		if (its_changed[dest[id]]) {
			n_changed[source[id]] = true;
		}
	}
}


void OptimizationPhase::optimize() {
	auto its_changed = thrust::device_vector<bool>(community.graph.n_nodes, false);

	for (int j = 0; j <= BUCKETS_RANGE.size(); j++) {
		#if PRINT_PERFORMANCE_LOG
			cudaEvent_t start, copy, sort, reduce_sort, transform, reduce_transform,  stop;
			cudaEventCreate(&start);
			cudaEventCreate(&copy);
			cudaEventCreate(&sort);
			cudaEventCreate(&reduce_sort);
			cudaEventCreate(&transform);
			cudaEventCreate(&reduce_transform);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
			std::cout << "Optimization Bucket " << j << std::endl;
		#endif

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

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(copy);
			cudaEventSynchronize(copy);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, copy);
			std::cout << "  - Copy Time : " << milliseconds << "ms" << std::endl;
		#endif

		int n_edge_in_buckets = p - selected_edge;
		key_community_source.resize(n_edge_in_buckets);
		key_community_dest.resize(n_edge_in_buckets);
		values_weight.resize(n_edge_in_buckets);

		if (n_edge_in_buckets == 0) {
			#if PRINT_DEBUG_LOG
				printf("No elements in bucket %d!\n", j);
			#endif 
			continue;
		}

		thrust::sort_by_key(
			key_community,
			key_community + n_edge_in_buckets,
			values_weight.begin()
		);

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(sort);
			cudaEventSynchronize(sort);
			cudaEventElapsedTime(&milliseconds, copy, sort);
			std::cout << " - Sort Time : " << milliseconds << "ms" << std::endl;
		#endif

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

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(reduce_sort);
			cudaEventSynchronize(reduce_sort);
+			cudaEventElapsedTime(&milliseconds, sort, reduce_sort);
			std::cout << " - Reduce after sort Time : " << milliseconds << "ms" << std::endl;
		#endif

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

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(transform);
			cudaEventSynchronize(transform);
			cudaEventElapsedTime(&milliseconds, reduce_sort, transform);
			std::cout << " - Transform Time : " << milliseconds << "ms" << std::endl;
		#endif
			

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

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(reduce_transform);
			cudaEventSynchronize(reduce_transform);
			cudaEventElapsedTime(&milliseconds, transform, reduce_transform);
			std::cout << " - Reduce after transform Time : " << milliseconds << "ms" << std::endl;
		#endif

		n_reduced_edges = ne.first - key_community_source.begin();

		int n_blocks = (n_reduced_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

		update_value_kernel << <n_blocks, BLOCK_SIZE >> > (
			n_reduced_edges,
			thrust::raw_pointer_cast(key_community_source.data()),
			thrust::raw_pointer_cast(key_community_dest.data()),
			thrust::raw_pointer_cast(reduced_value.data()),
			thrust::raw_pointer_cast(community.communities.data()),
			thrust::raw_pointer_cast(community.communities_weight.data()),
			thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
			thrust::raw_pointer_cast(its_changed.data())
			);

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, reduce_transform, stop);
			std::cout << " - Kernel Time : " << milliseconds << "ms" << std::endl;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "Total Optimization Time for Bucket "<< j <<" : " << milliseconds << "ms \n" << std::endl;
		#endif
	}

	int n_blocks = (community.graph.edge_source.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
	update_changed_kernel << <n_blocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(neighboorhood_change.data()),
		thrust::raw_pointer_cast(its_changed.data()),
		thrust::raw_pointer_cast(community.graph.edge_source.data()),
		thrust::raw_pointer_cast(community.graph.edge_destination.data()),
		community.graph.edge_source.size()
	);
}



#endif