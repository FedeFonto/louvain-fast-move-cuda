#ifndef GRAPH_DEVICE_H     
#define GRAPH_DEVICE_H

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>

#include "graphHost.cuh"


struct GraphDevice {

	unsigned int n_nodes = 0;
	unsigned long n_links = 0;
	double total_weight = 0;

	/* Data has been stored as compressed neighbor lists: every node is associated to an index in the degree lists.
	Degree at i postion contains the sum of node's degree from 0 to i-1.
	The weight assciated to a links in position i is in weight[i].
	*/
	 
	thrust::device_vector<unsigned int> n_of_neighboor;
	thrust::device_vector<float> tot_weight_per_nodes;
	thrust::device_vector<unsigned int> neighboorhood_sum;


	thrust::device_vector<unsigned int> edge_source;
	thrust::device_vector<unsigned int> edge_destination;
	thrust::device_vector<float> weights;


	GraphDevice(const GraphDevice& g) {
		n_nodes = g.n_nodes;
		n_links = g.n_links;
		total_weight = g.total_weight;
		edge_source = g.edge_source;
		edge_destination = g.edge_destination;
		weights = g.weights;
		tot_weight_per_nodes = g.tot_weight_per_nodes;
		n_of_neighboor = g.n_of_neighboor;
	};

	GraphDevice(GraphHost& g) {
		n_nodes = g.n_nodes;
		n_links = g.n_links;
		total_weight = g.total_weight;
		edge_source = g.edge_source;
		edge_destination = g.edge_destination;
		weights = g.weights;

		n_of_neighboor = thrust::device_vector<unsigned int>(n_nodes);
		tot_weight_per_nodes = thrust::device_vector<unsigned int>(n_nodes);
		neighboorhood_sum = thrust::device_vector<unsigned int>(n_nodes);

		auto key_community = thrust::make_zip_iterator(thrust::make_tuple(edge_source.begin(), edge_destination.begin()));

		thrust::sort_by_key(
			key_community,
			key_community + (n_links * 2),
			weights.begin()
		);

		thrust::reduce_by_key(
			edge_source.begin(),
			edge_source.end(),
			thrust::make_constant_iterator(1),
			thrust::make_discard_iterator(),
			n_of_neighboor.begin()
		);

		thrust::exclusive_scan(
			n_of_neighboor.begin(),
			n_of_neighboor.end(),
			neighboorhood_sum.begin() 
		);

		//todo fix for weighted
		thrust::copy(
			n_of_neighboor.begin(),
			n_of_neighboor.end(),
			tot_weight_per_nodes.begin()
		);
	};

	GraphDevice() {};
};

#endif