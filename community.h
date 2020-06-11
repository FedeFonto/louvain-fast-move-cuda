#ifndef COMMUNITY_h     
#define COMMUNITY_h

#include "graph.cuh"
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include "operators.h"
#include <thrust/transform_reduce.h>



typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator, thrust::device_vector<float>::iterator>> zip_pointer;


struct Community {
	GraphDevice graph;
	thrust::device_vector<unsigned int> communities;
	thrust::device_vector<double> communities_weight;

	thrust::device_vector<unsigned int> best_communities;

	zip_pointer start;
	zip_pointer end;

	unsigned int n_of_best_communities;
	double modularity = -0.5;
	cudaStream_t streams[2];


	
	Community(GraphHost& g) : graph(g) {
		cudaStreamCreate(&streams[0]);
		cudaStreamCreate(&streams[1]);
		n_of_best_communities = graph.n_nodes;
		communities = thrust::device_vector<unsigned int>(graph.n_nodes);
		communities_weight = thrust::device_vector<double>(graph.n_nodes);
		thrust::sequence(thrust::cuda::par.on(streams[0]), communities.begin(), communities.end(), 0);
		thrust::copy(thrust::cuda::par.on(streams[1]),  graph.tot_weight_per_nodes.begin(), graph.tot_weight_per_nodes.end(), communities_weight.begin());

		best_communities = thrust::device_vector<unsigned int>(graph.n_nodes);
		thrust::sequence(thrust::cuda::par.on(streams[0]), best_communities.begin(), best_communities.end(), 0);

		start = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.begin(), graph.edge_destination.begin(), graph.weights.begin()));
		end = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.end(), graph.edge_destination.end(), graph.weights.end()));
		compute_modularity();
	};

	void update_best(unsigned int* map) {
		thrust::transform(
			best_communities.begin(),
			best_communities.end(),
			best_communities.begin(),
			MakeCommunityBest(thrust::raw_pointer_cast(communities.data()), map)
		);
	}

	void update() {
		communities = thrust::device_vector<unsigned int>(graph.n_nodes);
		communities_weight = thrust::device_vector<float>(graph.n_nodes);
		thrust::sequence(communities.begin(), communities.end(), 0);
		thrust::copy(graph.tot_weight_per_nodes.begin(), graph.tot_weight_per_nodes.end(), communities_weight.begin());
		start = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.begin(), graph.edge_destination.begin(), graph.weights.begin()));
		end = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.end(), graph.edge_destination.end(), graph.weights.end()));
		compute_modularity();
	}


	void compute_modularity() {		
		double community_tot = thrust::transform_reduce(
			thrust::cuda::par.on(streams[0]),
			communities_weight.begin(),
			communities_weight.end(),
			Square<float>(),
			0,
			thrust::plus<float>()
		);

		auto values_weight = thrust::device_vector<float>(graph.n_links * 2);
		
		auto p = thrust::copy_if(
			thrust::cuda::par.on(streams[1]),
			graph.weights.begin(),
			graph.weights.end(),
			thrust::make_transform_iterator(start, MakeCommunityDest(thrust::raw_pointer_cast(communities.data()))),
			values_weight.begin(),
			ActualNeighboorhood(thrust::raw_pointer_cast(communities.data()))
		);

		double nodes_2_self_community = thrust::reduce(
			thrust::cuda::par.on(streams[1]),
			values_weight.begin(),
			values_weight.end()
		);
		
		modularity = nodes_2_self_community / (2 * graph.total_weight) - (community_tot)/ (4 * graph.total_weight * graph.total_weight);

	}
}; 

#endif
