#ifndef GRAPH_H     
#define GRAPH_H

#include <thrust/device_vector.h>


struct Graph {

	unsigned int n_nodes;
	unsigned long n_links;
	double total_weight;

	/* Data has been stored as compressed neighbor lists: every node is associated to an index in the degree lists.
	Degree at i postion contains the sum of node's degree from 0 to i-1.
	The weight assciated to a links in position i is in weight[i].
	*/
	thrust::device_vector<int> n_of_neighboor;
	thrust::device_vector<float> tot_weight_per_nodes;

	thrust::device_vector<int> edge_source;
	thrust::device_vector<int> edge_destination;
	thrust::device_vector<float> weights;

	Graph(){};
	Graph(std::string name, bool weighted);
};

#endif