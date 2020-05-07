#ifndef HASHMAP_H     
#define HASHMAP_H

#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include "constants.h"
#include "operators.h"
#include <thrust/partition.h>
#include <thrust/copy.h>

const int max_tentative = 100;

__device__
static inline int hash_position(unsigned long long l, int i, unsigned int size) {
	return (11*(i+1) * l + 17 * (i+1)) % 334214459  % size;
};


__device__
static bool insertHashTable(unsigned int k1,unsigned int k2, float v, unsigned int size, unsigned long long* pointer_k, float* pointer_v, int* conflict_stats) {
	unsigned long long l = (((unsigned long long) k1) << 32) | k2;
	bool f = true;

	for (int i = 0; i < max_tentative; i++) {
		atomicAdd(&conflict_stats[1], 1);
		int position = hash_position(l, i, size);
		unsigned long long check = atomicCAS(&pointer_k[position], FLAG, l);
		if (check == FLAG || check == l ) {
			atomicAdd(&pointer_v[position], v);
			return true;
		}
		else if(f){
			atomicAdd(conflict_stats, 1);
			f = false;
		}
	}
	atomicAdd(&conflict_stats[2], 1);
	return false;
};

__global__
static void kernel_optimization(unsigned int* k1,
								unsigned int* k2,
								float* v,
								int offset,
								int n_of_elem ,
								unsigned int size,
								unsigned long long* pointer_k1, 
								float* pointer_v, 
								int*conflict_stats, 
								float* self, 
								unsigned int* communities, 
								bool* community_change) 
{
	int id = offset + threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_of_elem) {
		if(community_change[k1[id]] && k1[id] != k2[id])
			if (communities[k1[id]] == communities[k2[id]])
				atomicAdd(&self[k1[id]], v[id]);
			else
				insertHashTable(k1[id], communities[k2[id]], v[id], size, pointer_k1, pointer_v, conflict_stats);
	}
};

__global__
static void kernel_aggregation(	unsigned int* k1,
								unsigned int* k2,
								float* v,
								int n_of_elem,
								unsigned int size,
								unsigned long long* pointer_k1,
								float* pointer_v,
								int* conflict_stats,
								unsigned int* communities,
								unsigned int* map)
{
	int id =  threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_of_elem) {
		insertHashTable(map[communities[k1[id]]]-1, map[communities[k2[id]]]-1, v[id], size, pointer_k1, pointer_v, conflict_stats);
	}
};


struct HashMap {	
	unsigned int size = BUCKETS_SIZE;

	thrust::device_vector<unsigned long long> key;
	thrust::device_vector<float> values;

	thrust::device_vector<float> self_community;
	thrust::device_vector<int> conflict_stats = thrust::device_vector<int>(3);

	unsigned long long* pointer_k;
	float* pointer_v;

	int* pointer_conflict_stats = thrust::raw_pointer_cast(conflict_stats.data());

	HashMap(unsigned int s) {
		size = s;
		key = thrust::device_vector<unsigned long long>(size, FLAG);
		values = thrust::device_vector<float>(size, 0);
		pointer_k = thrust::raw_pointer_cast(key.data());
		pointer_v = thrust::raw_pointer_cast(values.data());
	}

	void clear() {
		key.resize(size);
		values.resize(size);
		thrust::fill(key.begin(), key.end(), FLAG);
		thrust::fill(values.begin(), values.end(), 0);
		conflict_stats[0] = 0;
		conflict_stats[1] = 0;
		conflict_stats[2] = 0;
 	}

	void fill_for_optimization(	unsigned int* k1,
								unsigned int* k2,
								float* v, 
								int offset, 
								int n_of_elem, 
								int n_nodes,
								unsigned int* communities, 
								bool* community_change) {
		int n_blocks = (n_of_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
		clear();
		self_community = thrust::device_vector<float>(n_nodes);
		auto pointer_self_communities = thrust::raw_pointer_cast(self_community.data());
		kernel_optimization <<<n_blocks, BLOCK_SIZE >>> (k1, k2, v, offset, n_of_elem, size, pointer_k, pointer_v, pointer_conflict_stats, pointer_self_communities, communities, community_change);
	}


	void fill_for_aggregation(	unsigned int* k1,
								unsigned int* k2,
								float* v,
								int n_of_elem,
								unsigned int* communities,
								unsigned int* map){
		int n_blocks = (n_of_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
		clear();
		kernel_aggregation<<<n_blocks, BLOCK_SIZE >>> (k1, k2, v, n_of_elem, size, pointer_k, pointer_v, pointer_conflict_stats , communities, map);
	}

	int contract_array() {
		auto pair = thrust::make_zip_iterator(thrust::make_tuple(key.begin(), values.begin()));
		auto new_size = thrust::remove_if(pair, pair + size, key.begin(), Equals<unsigned long long>(FLAG));
		return new_size - pair;
	}

	

};

#endif
