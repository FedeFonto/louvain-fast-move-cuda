
#ifndef GRAPH_CU   
#define GRAPH_CU

#include "graph.cuh"

#include <iterator>
#include <fstream>
#include <set>
#include <vector>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>


using namespace std;

Graph::Graph() {};

Graph::Graph(std::string name, bool weighted ) {
	n_links = 0;

	ifstream f;
	f.open(name);

	int a, b;
	auto node_map = set<int>();

	printf("Start Parsing and Creation...\n");

	if (weighted) {
		//todo parsing 
	} else {
		while (f >> a >> b) {
			node_map.insert(a);
			node_map.insert(b);
			n_links++;
		}
	}

	n_nodes = node_map.size();
	f.clear();
	f.seekg(0, ios::beg);


	n_of_neighboor = thrust::device_vector<int>(n_nodes);
	tot_weight_per_nodes = thrust::device_vector<float>(n_nodes);

	auto temp_source = thrust::host_vector<int>(2 * n_links);
	auto temp_destination = thrust::host_vector<int>(2 * n_links);
	auto temp_weights = thrust::host_vector<float>(2 * n_links);

	int i = 0;
	if (weighted) {
		//todo parsing 
	}
	else {
		while (f >> a >> b) {
			temp_source[i] = a;
			temp_destination[i] = b;
			temp_weights[i] = 1;

			temp_source[i + 1] = b;
			temp_destination[i + 1] = a;
			temp_weights[i + 1] = 1;

			i += 2;
		}

		edge_source = thrust::device_vector<int>(temp_source);
		edge_destination = thrust::device_vector<int>(temp_destination);
		weights = thrust::device_vector<float>(temp_weights);

		auto key_community = thrust::make_zip_iterator(thrust::make_tuple(edge_source.begin(), edge_destination.begin()));


		thrust::sort_by_key(
			key_community,
			key_community + (n_links *2),
			weights.begin()
		);

		thrust::reduce_by_key(
			edge_source.begin(),
			edge_source.end(),
			thrust::make_constant_iterator(1),
			thrust::make_discard_iterator(),
			n_of_neighboor.begin()
		);

		thrust::copy(
			n_of_neighboor.begin(),
			n_of_neighboor.end(),
			tot_weight_per_nodes.begin()
		);

		total_weight = n_links;

	}
};

#endif
