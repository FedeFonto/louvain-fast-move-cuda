#ifndef OPERATORS_DELTA_H     
#define OPERATORS_DELTA_H

#include <thrust/functional.h>


struct DeltaModularityHash : public thrust::unary_function <thrust::tuple<unsigned long long, float>, float> {
	float* community_weight;
	float* nodes_weight;
	float total_weight;
	float* self_community;
	unsigned int* communities;

	__host__ __device__
		float operator()(thrust::tuple<unsigned long long, float> d) {
		const unsigned int node = (thrust::get<0>(d) >> 32);
		const unsigned int community = thrust::get<0>(d);

		float value = ((thrust::get<1>(d) - self_community[node]) / total_weight) +
			((nodes_weight[node] * (community_weight[communities[node]] - nodes_weight[node] - community_weight[community])) / (2 * total_weight * total_weight));
		return value;
	};

	DeltaModularityHash(float* w, float* n, float m, float* s, unsigned int* c) :
		community_weight(w), nodes_weight(n), total_weight(m), self_community(s), communities(c) {};
};


struct DeltaModularitysSort : public thrust::unary_function <thrust::tuple<unsigned int, unsigned int, float>, float> {
	float* community_weight;
	float* nodes_weight;
	float total_weight;
	float* self_community;
	unsigned int* communities;

	__host__ __device__
		float operator()(thrust::tuple<unsigned int, unsigned int, float> d) {
		float value = ((thrust::get<2>(d) - self_community[thrust::get<0>(d)]) / total_weight) +
			((nodes_weight[thrust::get<0>(d)] * (community_weight[communities[thrust::get<0>(d)]] - nodes_weight[thrust::get<0>(d)] - community_weight[thrust::get<1>(d)])) / (2 * total_weight * total_weight));
		return value;
	};

	DeltaModularitysSort(float* w, float* n, float m, float* s, unsigned int* c) :
		community_weight(w), nodes_weight(n), total_weight(m), self_community(s), communities(c) {};
};

#endif
