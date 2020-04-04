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
	thrust::device_vector<float> communities_weight;

	zip_pointer start;
	zip_pointer end;

	double modularity = -0.5;

	
	Community(GraphHost g) : graph(g) {
		communities = thrust::device_vector<unsigned int>(graph.n_nodes);
		communities_weight = thrust::device_vector<float>(graph.n_nodes);
		thrust::sequence(communities.begin(), communities.end(), 0);
		thrust::copy(graph.tot_weight_per_nodes.begin(), graph.tot_weight_per_nodes.end(), communities_weight.begin());
		start = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.begin(), graph.edge_destination.begin(), graph.weights.begin()));
		end = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.end(), graph.edge_destination.end(), graph.weights.end()));
		compute_modularity();
	};

	void update(GraphDevice* g) {
		graph = *g;
		communities = thrust::device_vector<unsigned int>(graph.n_nodes);
		communities_weight = thrust::device_vector<float>(graph.n_nodes);
		thrust::sequence(communities.begin(), communities.end(), 0);
		thrust::copy(graph.tot_weight_per_nodes.begin(), graph.tot_weight_per_nodes.end(), communities_weight.begin());
		start = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.begin(), graph.edge_destination.begin(), graph.weights.begin()));
		end = thrust::make_zip_iterator(thrust::make_tuple(graph.edge_source.end(), graph.edge_destination.end(), graph.weights.end()));
		compute_modularity();
	}


	void compute_modularity() {
		//todo optimize
		auto key_community_source = thrust::device_vector<unsigned int>(graph.n_links * 2);
		auto key_community_dest = thrust::device_vector<unsigned int>(graph.n_links * 2);
		auto values_weight = thrust::device_vector<float>(graph.n_links * 2);
		
		auto selected_edge = thrust::make_zip_iterator(thrust::make_tuple(key_community_source.begin(), key_community_dest.begin(), values_weight.begin()));

		auto p = thrust::copy_if(
			thrust::make_transform_iterator(start, MakeCommunityDest(thrust::raw_pointer_cast(communities.data()))),
			thrust::make_transform_iterator(end, MakeCommunityDest(thrust::raw_pointer_cast(communities.data()))),
			selected_edge,
			ActualNeighboorhood(thrust::raw_pointer_cast(communities.data()))
		);

		double nodes_2_self_community = thrust::reduce(
			values_weight.begin(),
			values_weight.end()
		);

		double community_tot = thrust::transform_reduce(
			communities_weight.begin(),
			communities_weight.end(),
			Square<float>(),
			0,
			thrust::plus<float>()
		);

		modularity = nodes_2_self_community / (2 * graph.total_weight) - (community_tot)/ (4 * graph.total_weight * graph.total_weight);

	}
}; 

#endif
