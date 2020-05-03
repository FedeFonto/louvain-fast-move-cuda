#ifndef HASHMAP_H     
#define HASHMAP_H

#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include "constants.h"
#include "operators.h"
#include <thrust/partition.h>
#include <thrust/copy.h>



__device__
static inline int hash_position(unsigned long long l, int i) {
	return (11*(i+1) * l + 17 * (i+1)) % 334214459  % BUCKETS_SIZE;
};


__device__
bool insertHashTable(unsigned int k1,unsigned int k2, float v, unsigned long long* pointer_k, float* pointer_v, int* c) {
	unsigned long long l = (((unsigned long long) k1) << 32) | k2;
	const int max_tentative = 1000;
	bool f = true;

	for (int i = 0; i < max_tentative; i++) {
		atomicAdd(&c[1], 1);
		int position = hash_position(l, i);
		unsigned long long check = atomicCAS(&pointer_k[position], FLAG, l);
		if (check == FLAG || check == l ) {
			atomicAdd(&pointer_v[position], v);
			return true;
		}
		else if(f){
			atomicAdd(c, 1);
			f = false;
		}
	}
	atomicAdd(&c[2], 1);
	return false;
};

__global__
void kernel(unsigned int* k1, unsigned int* k2, float* v, int n_of_elem , unsigned long long* pointer_k1, float* pointer_v, int*c) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_of_elem) {
		bool b = insertHashTable(k1[id], k2[id], v[id], pointer_k1, pointer_v, c);
	}
};



struct HashMap {	

	thrust::device_vector<unsigned long long> key = thrust::device_vector<unsigned long long>(BUCKETS_SIZE, FLAG);
	thrust::device_vector<float> values = thrust::device_vector<float>(BUCKETS_SIZE, 0);

	unsigned long long* pointer_k1 = thrust::raw_pointer_cast(key.data());
	float* pointer_v = thrust::raw_pointer_cast(values.data());
	thrust::device_vector<int> c = thrust::device_vector<int>(3);

	void fill(unsigned int* k1, unsigned int* k2, float* v, int n_of_elem) {
		int n_blocks = (n_of_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
		clear();
		kernel << <n_blocks, BLOCK_SIZE >> > (k1, k2, v, n_of_elem, pointer_k1, pointer_v, thrust::raw_pointer_cast(c.data()));
		std::cout << c[0] << ", " << (float)c[0] / n_of_elem << ", " << (float)c[1] / n_of_elem << ", " << c[2] << ",";
	}

	int resize() {
		auto pair = thrust::make_zip_iterator(thrust::make_tuple(key.begin(), values.begin()));
		auto new_size = thrust::remove_if(pair, pair + BUCKETS_SIZE, key.begin(), Equals<unsigned long long>(FLAG));
		return new_size - pair;
	}

	void clear() {
		key.resize(BUCKETS_SIZE);
		values.resize(BUCKETS_SIZE);
		thrust::fill(key.begin(), key.end(), FLAG);
		thrust::fill(values.begin(), values.end(), 0);
		c[0] = 0;
		c[1] = 0;
		c[2] = 0;
 	}

};

#endif
