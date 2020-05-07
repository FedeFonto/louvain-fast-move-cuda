#ifndef OPTIMIZATION_FAST_MOVE_CU     
#define OPTIMIZATION_FAST_MOVE_CU

#include "phaseOptimization.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
static void update_value_kernel_hash(
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
static void update_value_kernel_sort(
	unsigned int n_nodes,
	unsigned int* nodes_to_update,
	unsigned int* community_dest,
	float* delta_value,
	unsigned int* community,
	float* community_weight,
	float* nodes_weight,
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
			atomicAdd(&community_weight[c], nodes_weight[node]);
			atomicAdd(&community_weight[community[node]], nodes_weight[node] * -1);
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
	unsigned int n_edge
) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_edge) {
		if (its_changed[dest[id]]) {
			n_changed[source[id]] = true;
		}
	}
}


void OptimizationPhase::fast_move_update(const bool useHash) {
	thrust::fill(its_changed.begin(), its_changed.end(), false);
	thrust::fill(neighboorhood_change.begin(), neighboorhood_change.end(), false);
	int n_blocks;

	if (useHash) {
		n_blocks = (community.graph.n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
		update_value_kernel_hash << <n_blocks, BLOCK_SIZE >> > (
			community.graph.n_nodes,
			thrust::raw_pointer_cast(final_community.data()),
			thrust::raw_pointer_cast(final_value.data()),
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

	n_blocks = (community.graph.edge_source.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
	update_changed_kernel << <n_blocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(neighboorhood_change.data()),
		thrust::raw_pointer_cast(its_changed.data()),
		thrust::raw_pointer_cast(community.graph.edge_source.data()),
		thrust::raw_pointer_cast(community.graph.edge_destination.data()),
		community.graph.edge_source.size()
		);
}

#endif
