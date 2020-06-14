#ifndef OPERATORS_H     
#define OPERATORS_H

#include "constants.h"
#include "operatorsCommunity.h"
#include "operatorsDelta.h"
#include <thrust/functional.h>

struct CountNotZero : public thrust::unary_function<double, int> {
	__host__ __device__
	int operator()(double x) {
		return x > 0 ? 1 : 0;
	}

	CountNotZero() {};
};

template <typename T>
struct Equals : public thrust::unary_function<T, bool> {
	T value;
	__host__ __device__
		bool operator()(T x) {
		return x == value;
	}

	Equals(T v) : value(v) {};

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

template<typename T1, typename T2>
struct PairSum : public thrust::binary_function < thrust::tuple<T1, T2>, thrust::tuple<T1, T2>, thrust::tuple<T1, T2>>{

	__host__ __device__
	thrust::tuple<T1, T2> operator()(thrust::tuple<T1, T2> P1, thrust::tuple<T1, T2> P2) {
		return thrust::make_tuple(thrust::get<0>(P1) + thrust::get<0>(P2), thrust::get<1>(P1) + thrust::get<1>(P2));
	};
};

struct SplitIterator : public thrust::unary_function<thrust::tuple<unsigned long long, float>, thrust::tuple<unsigned int, unsigned int, float>> {
	__host__ __device__
	thrust::tuple<unsigned int, unsigned int, float> operator()(thrust::tuple<unsigned long long, float> d) {
		const unsigned int a = (thrust::get<0>(d) >> 32);
		const unsigned int b = thrust::get<0>(d);
		return thrust::make_tuple(a,b, thrust::get<1>(d));
	};

	SplitIterator() {};
};

template<typename T>
struct Square : public thrust::unary_function<T, T> {
	__host__ __device__
	double operator()(T d) {
		return d * d;
	};

	Square() {};
};

struct MapElement : public thrust::unary_function<unsigned int, unsigned int> {
	unsigned int* map;

	__host__ __device__
		unsigned int operator()(unsigned int d) {
		return map[d] - 1;
	};

	MapElement(unsigned int* v) : map(v) {};
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
		printf("\n GPU memory usage: used = %f, free = %f MB, total = %f MB\n\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

	};

	static std::string mode_name(MODE m) {
		if (m == SORT)
			return "SORT";
		if (m == HASH)
			return "HASH";
		if (m == ADAPTIVE_MEMORY)
			return "ADAPTIVE_MEMORY";
		if (m == ADAPTIVE_SPEED)
			return "ADAPTIVE_SPEED";
		return "UNKNOWN";
	}
};

#endif


