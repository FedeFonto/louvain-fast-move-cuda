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

__global__
void update_best_kernel(unsigned long long* key, float* value, unsigned int* best_community, float* best_values, int n_edges){
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_edges) {
		const unsigned int node = key[id] >> 32;
		const unsigned int community = key[id];
		if (atomicMaxFloat(&best_values[node], value[id]) < value[id]) {
			atomicExch(&best_community[node], community);
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
#if CSV_FORM
		std::cout << limit_round - round - hashmap->conflict_stats[3]<< "," << community.graph.edge_source.size() << "," << (float)(limit_round - round - hashmap->conflict_stats[3]) / community.graph.edge_source.size() * 100 << ",";
#else

		std::cout << "\nNumber of Edges selected: " << n_edge_in_buckets << " / " << community.graph.edge_source.size() << " (" << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << " %)" << std::endl;
		std::cout << " - Hashmap Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		int n_edge_in_buckets = hashmap->contract_array();


#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(resize);
		cudaEventSynchronize(resize);		
		cudaEventElapsedTime(&milliseconds, map, resize);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else

		std::cout << " - Resize Time : " << milliseconds << "ms" << std::endl;
#endif
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
#if CSV_FORM
		std::cout << milliseconds << ",";
#else

		std::cout << " - Delta Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		int n_blocks = (n_edge_in_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE;
		update_best_kernel << <n_blocks, BLOCK_SIZE >> > (
			hashmap-> pointer_k,
			hashmap-> pointer_v,
			thrust::raw_pointer_cast(final_community.data()),
			thrust::raw_pointer_cast(final_value.data()),
			n_edge_in_buckets
			);

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE
		cudaEventRecord(best);
		cudaEventSynchronize(best);
		cudaEventElapsedTime(&milliseconds, transform, best);
#if CSV_FORM
		std::cout << milliseconds <<std::endl;
#else

		std::cout << " - Best Selection Time : " << milliseconds << "ms" << std::endl;
#endif
#endif
		round = limit_round;
	}

}



#endif