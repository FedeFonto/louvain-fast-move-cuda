#include <thrust\device_vector.h>
#include <device_launch_parameters.h>
#include "constants.h"
#include "operators.h"
#include <thrust/copy.h>


__device__
static inline int hash_position(int k1, int k2, int i) {
	long long l =  i*(((long long) k1) << 32) | k2;
	return l % 334214459  % BUCKETS_SIZE;
};


__device__
bool insertHashTable(int k1, int k2, float v, int* pointer_k1, int* pointer_k2, float* pointer_v, int* c) {
	const int max_tentative = 1000;
	bool f = true;
	for (int i = 0; i < max_tentative; i++) {
		atomicAdd(&c[1], 1);
		int position = hash_position(k1, k2, i);
		int check = atomicCAS(&pointer_k2[position], FLAG, k2);
		if (check == FLAG) {
			atomicAdd(&pointer_k1[position], k1 + 1);
			atomicAdd(&pointer_v[position], v);
			return true;
		}
		else if (check == k2) {
			int check = atomicCAS(&pointer_k1[position], k1, k1);
			if (check == k1) {
				atomicAdd(&pointer_v[position], v);
				return true;
			}
		} else if(f){
			atomicAdd(c, 1);
			f = false;
		}
	}
	atomicAdd(&c[2], 1);
	return false;
};

__global__
void kernel(unsigned int* k1, unsigned int* k2, float* v, int n_of_elem , int* pointer_k1, int* pointer_k2, float* pointer_v, int*c) {
	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < n_of_elem) {
		bool b = insertHashTable((int)k1[id], (int)k2[id], v[id], pointer_k1, pointer_k2, pointer_v, c);
	}
};



struct HashMap {	

	thrust::device_vector<int> key_1 = thrust::device_vector<int>(BUCKETS_SIZE, FLAG);
	thrust::device_vector<int> key_2 = thrust::device_vector<int>(BUCKETS_SIZE, FLAG);
	thrust::device_vector<float> values = thrust::device_vector<float>(BUCKETS_SIZE, 0);
	int* pointer_k1 = thrust::raw_pointer_cast(key_1.data());
	int* pointer_k2 = thrust::raw_pointer_cast(key_2.data());
	float* pointer_v = thrust::raw_pointer_cast(values.data());
	thrust::device_vector<int> c = thrust::device_vector<int>(3);

	void fill(unsigned int* k1, unsigned int* k2, float* v, int n_of_elem) {
		int n_blocks = (n_of_elem + BLOCK_SIZE - 1) / BLOCK_SIZE;
		c[0] = 0;
		c[1] = 0;
		c[2] = 0;
		clear();
		kernel << <n_blocks, BLOCK_SIZE >> > (k1, k2, v, n_of_elem, pointer_k1, pointer_k2, pointer_v, thrust::raw_pointer_cast(c.data()));
		std::cout << "C: " << c[0] << " C%: " << (float) c[0]/ n_of_elem << " mean n of search: " << (float) c[1]/n_of_elem << " Errors: " << c[2] ;
	}
	
	void remove_void() {
		thrust::device_vector<int> k1 = thrust::device_vector<int>(BUCKETS_SIZE);
		thrust::device_vector<int> k2 = thrust::device_vector<int>(BUCKETS_SIZE);
		thrust::device_vector<float> v = thrust::device_vector<float>(BUCKETS_SIZE);

		auto out = thrust::make_zip_iterator(thrust::make_tuple(k1.begin(), k2.begin(), v.begin()));
		auto in = thrust::make_zip_iterator(thrust::make_tuple(key_1.begin(), key_2.begin(), values.begin()));
		thrust::copy_if(in, in + SIZE_MAX, key_1.begin(), out, NotNegative());
	}

	void clear() {
		thrust::fill(key_1.begin(), key_1.end(), FLAG);
		thrust::fill(key_2.begin(), key_2.end(), FLAG);
		thrust::fill(values.begin(), values.end(), 0);
	}

};
