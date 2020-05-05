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

__global__
void update_value_kernel(
	unsigned int n_nodes,
	unsigned int* community_dest,
	float* delta_value,
	unsigned int* community,
	float* community_weight,
	float* nodes_weight,
	bool* its_changed
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_nodes) {
		int node = id;
		int c = community_dest[id];
		if (community[node] == c || delta_value[id] <= 0) {
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
	unsigned int* source,
	unsigned int* dest,
	unsigned int n_edge
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_edge) {
		if (its_changed[dest[id]]) {
			n_changed[source[id]] = true;
		}
	}
}

void OptimizationPhase::optimize() {
#if PRINT_PERFORMANCE_LOG
	cudaEvent_t start, round_start, copy, map, resize, transform, best, kernel_update, stop;
	cudaEventCreate(&round_start);
	cudaEventCreate(&start);
	cudaEventCreate(&copy);
	cudaEventCreate(&map);
	cudaEventCreate(&resize);
	cudaEventCreate(&transform);
	cudaEventCreate(&best);
	cudaEventCreate(&kernel_update);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	float milliseconds = 0;
#endif
	auto its_changed = thrust::device_vector<bool>(community.graph.n_nodes, false);

	int limit_round;
	int round = 0;

	while (round < community.graph.edge_destination.size()) {
#if PRINT_PERFORMANCE_LOG

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

		int n_edge_in_buckets = limit_round;

#if PRINT_PERFORMANCE_LOG
		cudaEventRecord(copy);
		cudaEventSynchronize(copy);
		cudaEventElapsedTime(&milliseconds, round_start, copy);
#if CSV_FORM
		std::cout << n_edge_in_buckets << "," << community.graph.edge_source.size() << "," << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << ",";
#else

		std::cout << "\nNumber of Edges selected: " << n_edge_in_buckets << " / " << community.graph.edge_source.size() << " (" << (float)n_edge_in_buckets / community.graph.edge_source.size() * 100 << " %)" << std::endl;
		std::cout << " - Copy Time : " << milliseconds << "ms" << std::endl;
#endif
#endif
		h.fill_for_optimization(	
				thrust::raw_pointer_cast(community.graph.edge_source.data()),
				thrust::raw_pointer_cast(community.graph.edge_destination.data()),
				thrust::raw_pointer_cast(community.graph.weights.data()),
				round,
				n_edge_in_buckets,
				community.graph.n_nodes,
				thrust::raw_pointer_cast(community.communities.data()), 
				thrust::raw_pointer_cast(neighboorhood_change.data())
			);

#if PRINT_PERFORMANCE_LOG
		cudaEventRecord(map);
		cudaEventSynchronize(map);
		cudaEventElapsedTime(&milliseconds, copy, map);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else

		std::cout << " - Hashmap Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		n_edge_in_buckets = h.resize();


#if PRINT_PERFORMANCE_LOG
		cudaEventRecord(resize);
		cudaEventSynchronize(resize);
		cudaEventElapsedTime(&milliseconds, map, resize);
#if CSV_FORM
		std::cout << milliseconds << ",";
#else

		std::cout << " - Resize Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

		auto pair = thrust::make_zip_iterator(thrust::make_tuple(h.key.begin(), h.values.begin()));
		
		thrust::transform(
			pair,
			pair + n_edge_in_buckets,
			h.values.begin(),
			DeltaModularity(
				thrust::raw_pointer_cast(community.communities_weight.data()),
				thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
				community.graph.total_weight,
				thrust::raw_pointer_cast(h.self_community.data()),
				thrust::raw_pointer_cast(community.communities.data())
			)
		);



#if PRINT_PERFORMANCE_LOG
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
			h.pointer_k,
			h.pointer_v,
			thrust::raw_pointer_cast(final_community.data()),
			thrust::raw_pointer_cast(final_value.data()),
			n_edge_in_buckets
			);

#if PRINT_PERFORMANCE_LOG
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

	int n_blocks = (community.graph.n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;


	update_value_kernel << <n_blocks, BLOCK_SIZE >> > (
		community.graph.n_nodes,
		thrust::raw_pointer_cast(final_community.data()),
		thrust::raw_pointer_cast(final_value.data()),
		thrust::raw_pointer_cast(community.communities.data()),
		thrust::raw_pointer_cast(community.communities_weight.data()),
		thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
		thrust::raw_pointer_cast(its_changed.data())
		);

#if PRINT_PERFORMANCE_LOG
	cudaEventRecord(kernel_update);
	cudaEventSynchronize(kernel_update);
	cudaEventElapsedTime(&milliseconds, best, kernel_update);
#if CSV_FORM
	std::cout << milliseconds << ",";
#else
	std::cout << " - Kernel Update Time : " << milliseconds << "ms" << std::endl;
#endif
#endif

	n_blocks = (community.graph.edge_source.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

	thrust::fill(neighboorhood_change.begin(), neighboorhood_change.end(), false);


	update_changed_kernel << <n_blocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(neighboorhood_change.data()),
		thrust::raw_pointer_cast(its_changed.data()),
		thrust::raw_pointer_cast(community.graph.edge_source.data()),
		thrust::raw_pointer_cast(community.graph.edge_destination.data()),
		community.graph.edge_source.size()
		);


#if PRINT_PERFORMANCE_LOG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, kernel_update, stop);
#if CSV_FORM
	std::cout << milliseconds << ",";
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << milliseconds << std::endl;

#else
	std::cout << " - Kernel Changed Time : " << milliseconds << "ms" << std::endl;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Optimization Time: " << milliseconds << "ms \n" << std::endl;
#endif
#endif
}



#endif