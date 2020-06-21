#ifndef OPTIMIZATION_FAST_MOVE_CU     
#define OPTIMIZATION_FAST_MOVE_CU

#include "phaseOptimization.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __host__
float float_from_comparable_integer(int f) {
	return *reinterpret_cast<float*>(&f);
}

__global__
static void update_value_kernel_hash(
	unsigned int n_nodes,
	unsigned long long* community_pair,
	unsigned int* community,
	double* community_weight,
	unsigned* nodes_weight,
	bool* its_changed
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_nodes) {
		int node = id;
		unsigned long long key = community_pair[node];
		float delta_value = float_from_comparable_integer(key >> 32);
		const unsigned int c = key;
		if (community[node] == c || delta_value <= 0) {
			return;
		}
		else {
			double v = nodes_weight[node];
			atomicAdd(&community_weight[c], v);
			atomicAdd(&community_weight[community[node]], v* -1);
			community[node] = c;
			its_changed[node] = true;
		}
	}
};

__global__
static void update_value_kernel_sort(
	unsigned int n_nodes,
	unsigned int* nodes_to_update,
	unsigned int* community_dest,
	float* delta_value,
	unsigned int* community,
	double* community_weight,
	unsigned* nodes_weight,
	bool* its_changed
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_nodes) {
		int node = nodes_to_update[id];
		int c = community_dest[id];
		if (community[node] == c || delta_value[id] <= 0) {
			return;
		}
		else {
			double v = nodes_weight[node];
			atomicAdd(&community_weight[c], v);
			atomicAdd(&community_weight[community[node]], v * -1);
			community[node] = c;
			its_changed[node] = true;
		}
	}
};

__global__
static void update_value_kernel_fast(
	unsigned int n_nodes,
	unsigned int* nodes_to_update,
	unsigned int* community_dest,
	float* delta_value,
	unsigned int* community,
	double* community_weight,
	unsigned* nodes_weight,
	bool* its_changed
) {

	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_nodes) {
		int node = nodes_to_update[id];
		int c = community_dest[id];
		if (community[node] == c || delta_value[id] <= 0 || node > c) {
			return;
		}
		else {
			double v = nodes_weight[node];
			atomicAdd(&community_weight[c], v);
			atomicAdd(&community_weight[community[node]], v * -1);
			community[node] = c;
			its_changed[node] = true;
		}
	}
};

__global__
static void update_changed_kernel(
	bool* n_changed,
	bool* its_changed,
	unsigned int* source,
	unsigned int* dest,
	unsigned int n_edge,
	unsigned int* communities
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_edge) {
		if (its_changed[dest[id]] && communities[dest[id]] != communities[source[id]]) {
			n_changed[source[id]] = true;
		}
	}
}


void OptimizationPhase::fast_move_update(const bool useHash) {
#if  PRINT_PERFORMANCE_LOG && INCLUDE_UPDATES
	cudaEvent_t a, b;
	cudaEventCreate(&a);
	cudaEventCreate(&b);
	cudaEventRecord(a);
#endif
	thrust::fill(its_changed.begin(), its_changed.end(), false);
	thrust::fill(neighboorhood_change.begin(), neighboorhood_change.end(), false);
	int n_blocks;
	if (execution_number == 0) {
		n_blocks = (nodes_considered + BLOCK_SIZE - 1) / BLOCK_SIZE;
		update_value_kernel_fast<< <n_blocks, BLOCK_SIZE >> > (
			nodes_considered,
			thrust::raw_pointer_cast(final_node.data()),
			thrust::raw_pointer_cast(final_community.data()),
			thrust::raw_pointer_cast(final_value.data()),
			thrust::raw_pointer_cast(community.communities.data()),
			thrust::raw_pointer_cast(community.communities_weight.data()),
			thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
			thrust::raw_pointer_cast(its_changed.data())
			);
	} else if (useHash) {
		n_blocks = (community.graph.n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
		update_value_kernel_hash << <n_blocks, BLOCK_SIZE >> > (
			community.graph.n_nodes,
			thrust::raw_pointer_cast(final_pair.data()),
			thrust::raw_pointer_cast(community.communities.data()),
			thrust::raw_pointer_cast(community.communities_weight.data()),
			thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
			thrust::raw_pointer_cast(its_changed.data())
			);
	}
	else {
		n_blocks = (nodes_considered + BLOCK_SIZE - 1) / BLOCK_SIZE;
		update_value_kernel_sort << <n_blocks, BLOCK_SIZE >> > (
			nodes_considered,
			thrust::raw_pointer_cast(final_node.data()),
			thrust::raw_pointer_cast(final_community.data()),
			thrust::raw_pointer_cast(final_value.data()),
			thrust::raw_pointer_cast(community.communities.data()),
			thrust::raw_pointer_cast(community.communities_weight.data()),
			thrust::raw_pointer_cast(community.graph.tot_weight_per_nodes.data()),
			thrust::raw_pointer_cast(its_changed.data())
			);
	}

#if PRINT_PERFORMANCE_LOG && INCLUDE_UPDATES
	cudaEventRecord(b);
	cudaEventSynchronize(b);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, a, b);
	std::cout << milliseconds << ",";

#endif

	n_blocks = (community.graph.edge_source.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

#if PRINT_PERFORMANCE_LOG && INCLUDE_UPDATES
	cudaEventRecord(a);
	cudaEventSynchronize(a);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, b, a);
	std::cout << milliseconds << std::endl;

#endif
}

#endif
