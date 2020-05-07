#ifndef OPERATORS_C_H     
#define OPERATORS_C_H

#include <thrust/functional.h>


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

struct MakeCommunityDest : public thrust::unary_function<thrust::tuple<unsigned int, unsigned int, float>, thrust::tuple<unsigned int, unsigned int, float>> {
	unsigned int* values;

	__host__ __device__
		thrust::tuple<unsigned int, unsigned int, float> operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		return thrust::make_tuple<int, int, float>(
			thrust::get<0>(d),
			values[thrust::get<1>(d)],
			thrust::get<2>(d)
			);
	};

	MakeCommunityDest(unsigned int* v) : values(v) {};
};


struct MakeCommunityPair : public thrust::unary_function<thrust::tuple<unsigned int, unsigned int, float>, thrust::tuple<unsigned int, unsigned int, float>> {
	unsigned int* communities;
	unsigned int* map;

	__host__ __device__
		thrust::tuple<unsigned int, unsigned int, float> operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		return thrust::make_tuple<int, int, float>(
			map[communities[thrust::get<0>(d)]] - 1,
			map[communities[thrust::get<1>(d)]] - 1,
			thrust::get<2>(d)
			);
	};

	MakeCommunityPair(unsigned int* c, unsigned int* v) : communities(c), map(v) {};
};


struct MakeCommunityBest : public thrust::unary_function<unsigned int, unsigned int> {
	unsigned int* map;
	unsigned int* communities;

	__host__ __device__
		unsigned int operator()(unsigned int d) {
		return map[communities[d]] - 1;
	};

	MakeCommunityBest(unsigned int* c, unsigned int* v) : communities(c), map(v) {};
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

	ActualNeighboorhood(unsigned int* c) : communities(c) {}
};

#endif