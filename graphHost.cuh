#ifndef GRAPH_HOST_H    
#define GRAPH_HOST_H

#include <thrust/host_vector.h>


struct GraphHost {
	unsigned int n_nodes;
	unsigned long n_links;
	double total_weight;
	unsigned int min_id;

	thrust::host_vector<unsigned int> edge_source;
	thrust::host_vector<unsigned int> edge_destination;
	thrust::host_vector<float> weights;

	GraphHost(std::string name, bool weighted, int skip_line);
};

#endif