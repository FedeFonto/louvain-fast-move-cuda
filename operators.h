#ifndef OPERATORS_H     
#define OPERATORS_H

#include <thrust/functional.h>

struct CountNotZero : public thrust::unary_function<float, int> {
	__host__ __device__
	int operator()(float x) {
		return x > 0 ? 1 : 0;
	}

	CountNotZero() {};
};

struct InWitchBucket : public thrust::unary_function<int, int> {
	const int* range;
	const int size;

	__host__ __device__
	int operator()(int x) {
		for (int i = 0; i < size - 1;  i++) {
			if (range[i] <= x && x < range[i + 1]) {
				return i;
			}
		}
		return size - 1;
	};

	InWitchBucket(const int* r, const int s) : range(r), size(s) {};
};


struct TestTupleValue : public thrust::unary_function<int, bool> {
	int condition; 
	int* buckets;
	bool* changed;

	__host__ __device__
	bool operator()(int t) {
		bool test = condition == buckets[t] && changed[t];
		if (test) {
			changed[t] = false;
		}
		return test;
	};

	TestTupleValue(int* b, bool* c, int p) : buckets(b), changed(c) , condition(p) {};
};


struct MakeCommunityDest : public thrust::unary_function<thrust::tuple<int, int, float>, thrust::tuple<int, int, float>>{
	int*  values;

	__host__ __device__
	thrust::tuple<int, int, float> operator()(thrust::tuple<int, int, float> d) {
		return thrust::make_tuple<int,int,float>(
			thrust::get<0>(d),
			values[thrust::get<1>(d)],
			thrust::get<2>(d)
		);
	};

	MakeCommunityDest(int* v) : values(v){};
};


struct MakeCommunityPair : public thrust::unary_function<thrust::tuple<int, int, float>, thrust::tuple<int, int, float>> {
	int* communities;
	int* map;

	__host__ __device__
		thrust::tuple<int, int, float> operator()(thrust::tuple<int, int, float> d) {
		return thrust::make_tuple<int, int, float>(
			map[communities[thrust::get<0>(d)] ]-1,
			map[communities[thrust::get<1>(d)] ]-1,
			thrust::get<2>(d)
			);
	};

	MakeCommunityPair(int* c, int* v) : communities(c), map(v) {};
};


struct GetMaxValue : public thrust::binary_function<thrust::tuple<int, float>, thrust::tuple<int, float>, thrust::tuple<int, float>> {
	__host__ __device__
	thrust::tuple<int, float> operator()(thrust::tuple<int, float> a, thrust::tuple<int, float> b ) {
		if (thrust::get<1>(b) > thrust::get<1>(a) 
			|| (thrust::get<1>(b) == thrust::get<1>(a) && thrust::get<0>(a) > thrust::get<0>(b)))
			return b;
		else
			return a;
	};

	GetMaxValue(){};
};

struct DeltaModularity : public thrust::unary_function <thrust::tuple<int, int, float>, float> {
	float* community_weight;
	float* nodes_weight;
	float total_weight;

	__host__ __device__
	float operator()(thrust::tuple<int, int, float> d) {
		float value = ((thrust::get<2>(d)) / total_weight) +
			(( nodes_weight[thrust::get<0>(d)] * -1 * community_weight[thrust::get<1>(d)]) / (2 * total_weight * total_weight));
		return value;
	};

	DeltaModularity(float* w, float *n, float m) :
		community_weight(w),nodes_weight(n),total_weight(m) {};
};

struct ActualNeighboorhood : public thrust::unary_function <thrust::tuple<int, int, float>, bool> {
	int* communities;

	__host__ __device__
	bool operator()(thrust::tuple<int, int, float> d) {
		if (communities[thrust::get<0>(d)] == thrust::get<1>(d)) {
			return true;
		}
		return false;
	};

	ActualNeighboorhood(int* c) : communities(c){}
};

struct ActualNeighboorhoodCopy : public thrust::unary_function <thrust::tuple<int, int, float>, bool> {
	int* communities;
	float* values;

	__host__ __device__
		bool operator()(thrust::tuple<int, int, float> d) {
		if (communities[thrust::get<0>(d)] == thrust::get<1>(d)) {
			values[thrust::get<0>(d)] = thrust::get<2>(d);
			return true;
		}
		return false;
	};

	ActualNeighboorhoodCopy(int* c, float* v) : communities(c), values(v) {}
};

struct MyUtils {
	static void print_memory_usage() {
		// show memory usage of GPU
		size_t free_byte;
		size_t total_byte;
		auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
		if (cudaSuccess != cuda_status) {
			printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
			exit(1);
		}

		double free_db = (double)free_byte;
		double total_db = (double)total_byte;
		double used_db = total_db - free_db;
		printf("\n GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

	};
};

#endif


