#ifndef GRAPH_DEVICE_H     
#define GRAPH_DEVICE_H

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>

#include "constants.h"
#include "operators.h"
#include "graphHost.cuh"

__global__
static void renumber_kernel(
	unsigned int* k,
	unsigned int* map,
	unsigned int limit) {

	int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (id < limit) {
		map[k[id]] = 1;
	}
}

struct GraphDevice {

	unsigned int n_nodes = 0;
	unsigned long n_links = 0;
	double total_weight = 0;

	/* Data has been stored as compressed neighbor lists: every node is associated to an index in the degree lists.
	Degree at i postion contains the sum of node's degree from 0 to i-1.
	The weight assciated to a links in position i is in weight[i].
	*/
	 
	thrust::device_vector<unsigned int> n_of_neighboor;
	thrust::device_vector<unsigned> tot_weight_per_nodes;
	thrust::device_vector<unsigned int> neighboorhood_sum;


	thrust::device_vector<unsigned int> edge_source;
	thrust::device_vector<unsigned int> edge_destination;
	thrust::device_vector<unsigned int> weights;


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

		if (g.max_id - g.min_id >= n_nodes) {
			std::cout << "Renumbering" << std::endl;
			auto map = thrust::device_vector<unsigned int>(g.max_id + 1, 0);
			int n_blocks = (edge_source.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

			renumber_kernel<<<n_blocks, BLOCK_SIZE>>>(
				thrust::raw_pointer_cast(edge_source.data()),
				thrust::raw_pointer_cast(map.data()),
				edge_source.size()
				);

			//prefix sum
			thrust::inclusive_scan(
				map.begin(),
				map.end(),
				map.begin()
			);

			thrust::transform(
				edge_source.begin(),
				edge_source.end(),
				edge_source.begin(),
				MapElement(thrust::raw_pointer_cast(map.data()))
			);

			thrust::transform(
				edge_destination.begin(),
				edge_destination.end(),
				edge_destination.begin(),
				MapElement(thrust::raw_pointer_cast(map.data()))
			);
		}
		else {
			thrust::transform(edge_source.begin(),
				edge_source.end(),
				thrust::make_constant_iterator(g.min_id),
				edge_source.begin(),
				thrust::minus<int>());

			thrust::transform(edge_destination.begin(),
				edge_destination.end(),
				thrust::make_constant_iterator(g.min_id),
				edge_destination.begin(),
				thrust::minus<int>());
		}

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