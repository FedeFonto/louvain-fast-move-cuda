#ifndef OPTIMIZATION_CU     
#define OPTIMIZATION_CU

#include "phaseOptimization.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ 
float atomicMaxFloat(float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

	return old;
}
__device__ __host__
unsigned int float_to_comparable_integer(float f) {
	return *reinterpret_cast<unsigned int*>(&f);
}


__global__
void update_best_kernel(unsigned long long* key, float* value, unsigned long long int* best_community, int n_edges){
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_edges) {
		unsigned int community = key[id];
		unsigned int node = key[id] >> 32;
		if (value[id] > 0) {
			auto k = (((unsigned long long) float_to_comparable_integer(value[id])) << 32 | community);
			atomicMax(&best_community[node], k);
		}
		
	}
}


void OptimizationPhase::optimize_hash() {
#if PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
	cudaEvent_t  round_start, map, resize, transform, best, kernel_update, stop;
	cudaEventCreate(&round_start);
	cudaEventCreate(&map);
	cudaEventCreate(&resize);
	cudaEventCreate(&transform);
	cudaEventCreate(&best);
	cudaEventCreate(&kernel_update);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	unsigned analyzed = 0;
	unsigned access_num = 0;
	unsigned access_dem = 0;
	float map_sum = 0;
	float resize_sum = 0;
	float transform_sum = 0;
	float best_sum = 0;
	float update_sum = 0;
	float total = 0;
#endif

	int limit_round;
	int round = 0;

	while (round < community.graph.edge_destination.size()) {
#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(round_start);
#endif
	
		limit_round = round + STEP_ROUND;
		if (limit_round >= community.graph.edge_destination.size()) {
			limit_round = community.graph.edge_destination.size();
		}
		else {
			limit_round = community.graph.neighboorhood_sum[community.graph.edge_source[limit_round] - 1];
		}

		hashmap->fill_for_optimization(	
				thrust::raw_pointer_cast(community.graph.edge_source.data()),
				thrust::raw_pointer_cast(community.graph.edge_destination.data()),
				thrust::raw_pointer_cast(community.graph.weights.data()),
				round,
				limit_round,
				community.graph.n_nodes,
				thrust::raw_pointer_cast(community.communities.data()), 
				thrust::raw_pointer_cast(neighboorhood_change.data())
			);

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(map);
		cudaEventSynchronize(map);
		cudaEventElapsedTime(&milliseconds, round_start, map);
		analyzed += hashmap->conflict_stats[3];
		access_num += hashmap->conflict_stats[1];
		access_dem += hashmap->conflict_stats[0];
		map_sum += milliseconds;
#endif

		int n_edge_in_buckets = hashmap->contract_array();


#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(resize);
		cudaEventSynchronize(resize);		
		cudaEventElapsedTime(&milliseconds, map, resize);
		resize_sum += milliseconds;
#endif

		auto pair = thrust::make_zip_iterator(thrust::make_tuple(hashmap->key.begin(), hashmap->values.begin()));
		
		thrust::transform(
			pair,
			pair + n_edge_in_buckets,
			hashmap->values.begin(),
			DeltaModularityHash(
				thrust::raw_pointer_cast(community.communities_weight.data()),
				thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
				community.graph.total_weight,
				thrust::raw_pointer_cast(hashmap->self_community.data()),
				thrust::raw_pointer_cast(community.communities.data())
			)
		);



#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(transform);
		cudaEventSynchronize(transform);
		cudaEventElapsedTime(&milliseconds, resize, transform);
		transform_sum += milliseconds;
#endif
	
		thrust::max_element(hashmap->values.begin(), hashmap->values.begin() + n_edge_in_buckets);
		int n_blocks = (n_edge_in_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE;
		update_best_kernel << <n_blocks, BLOCK_SIZE >> > (
			hashmap-> pointer_k,
			hashmap-> pointer_v,
			thrust::raw_pointer_cast(final_pair.data()),
			n_edge_in_buckets
			);

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(best);
		cudaEventSynchronize(best);
		cudaEventElapsedTime(&milliseconds, transform, best);
		update_sum += milliseconds;
		cudaEventElapsedTime(&milliseconds, round_start, best);
		total += milliseconds;
#endif
		round = limit_round;
	}

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
	std::cout << community.graph.edge_destination.size() << ",";
	std::cout << community.graph.edge_destination.size() - analyzed << ",";
	std::cout << (float) (community.graph.edge_destination.size() - analyzed) / community.graph.edge_destination.size() << ",";
	std::cout << (float) access_num / access_dem << ",";
	std::cout << map_sum << ",";
	std::cout << resize_sum << ",";
	std::cout << transform_sum << ",";
	std::cout << best_sum << ",";
	std::cout << update_sum << ",";
	std::cout << total << std::endl;
#endif
}



#endif