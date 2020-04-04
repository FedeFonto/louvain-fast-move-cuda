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

struct TestTupleValue : public thrust::unary_function<thrust::tuple<unsigned int, unsigned int>, bool> {
	bool* changed;

	__host__ __device__
	bool operator()(thrust::tuple<unsigned int, unsigned int> d) {
		unsigned int source = thrust::get<0>(d);
		unsigned int dest = thrust::get<1>(d);

		return changed[source] && source != dest;
	};

	TestTupleValue(bool* c) : changed(c) {};
};


struct MakeCommunityDest : public thrust::unary_function<thrust::tuple<unsigned int, unsigned int, float>, thrust::tuple<unsigned int, unsigned int, float>>{
	unsigned int*  values;

	__host__ __device__
		thrust::tuple<unsigned int, unsigned int, float> operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		return thrust::make_tuple<int,int,float>(
			thrust::get<0>(d),
			values[thrust::get<1>(d)],
			thrust::get<2>(d)
		);
	};

	MakeCommunityDest(unsigned int* v) : values(v){};
};


struct MakeCommunityPair : public thrust::unary_function<thrust::tuple<unsigned int,unsigned int, float>, thrust::tuple<unsigned int, unsigned int, float>> {
	unsigned int* communities;
	unsigned int* map;

	__host__ __device__
		thrust::tuple<unsigned int, unsigned int, float> operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		return thrust::make_tuple<int, int, float>(
			map[communities[thrust::get<0>(d)] ]-1,
			map[communities[thrust::get<1>(d)] ]-1,
			thrust::get<2>(d)
			);
	};

	MakeCommunityPair(unsigned int* c, unsigned int* v) : communities(c), map(v) {};
};


struct GetMaxValue : public thrust::binary_function<thrust::tuple<unsigned int, float>, thrust::tuple<unsigned int, float>, thrust::tuple<unsigned int, float>> {
	__host__ __device__
	thrust::tuple<unsigned int, float> operator()(thrust::tuple<unsigned int, float> a, thrust::tuple<unsigned int, float> b ) {
		if (thrust::get<1>(b) > thrust::get<1>(a))
			return b;
		else
			return a;
	};

	GetMaxValue(){};
};

struct DeltaModularity : public thrust::unary_function <thrust::tuple<unsigned int, unsigned int, float>, float> {
	float* community_weight;
	float* nodes_weight;
	float total_weight;
	float* self_community;
	unsigned int* communities;

	__host__ __device__
	float operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		float value = ((thrust::get<2>(d) - self_community[thrust::get<0>(d)]) / total_weight) +
			(( nodes_weight[thrust::get<0>(d)] * (community_weight[communities[thrust::get<0>(d)]] - nodes_weight[thrust::get<0>(d)] - community_weight[thrust::get<1>(d)])) / (2 * total_weight * total_weight));
		return value;
	};

	DeltaModularity(float* w, float *n, float m, float* s,unsigned int* c) :
		community_weight(w),nodes_weight(n),total_weight(m),self_community(s), communities(c){};
};

struct ActualNeighboorhood : public thrust::unary_function <thrust::tuple<unsigned int, unsigned int, float>, bool> {
	unsigned int* communities;

	__host__ __device__
	bool operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		if (communities[thrust::get<0>(d)] == thrust::get<1>(d)) {
			return true;
		}
		return false;
	};

	ActualNeighboorhood(unsigned int* c) : communities(c){}
};

template<typename T>
struct Square : public thrust::unary_function<T, T> {
	__host__ __device__
	T operator()(T d) {
		return d * d;
	};

	Square() {};
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


