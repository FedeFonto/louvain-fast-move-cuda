
#ifndef GRAPH_CU   
#define GRAPH_CU

#include "graphHost.cuh"
#include "operators.h"

#include <iterator>
#include <fstream>
#include <set>
#include <vector>


using namespace std;


GraphHost::GraphHost(std::string name, bool weighted ) {
	n_links = 0;

	ifstream f;
	f.open(name);

	int a, b;
	auto node_map = set<int>();

	printf("Start Parsing...\n");

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

	if (n_nodes == 0) {
		std::cout << "Error during parsing: no node found" << std::endl;
		exit(0);
	}

	printf("N of edges and links obtained\n");

	edge_source = thrust::host_vector<int>(2 * n_links);
	edge_destination = thrust::host_vector<int>(2 * n_links);
	weights = thrust::host_vector<float>(2 * n_links);

	int i = 0;
	if (weighted) {
		//todo parsing 
	}
	else {
		while (f >> a >> b) {
			edge_source[i] = a;
			edge_destination[i] = b;
			weights[i] = 1;

			edge_source[i + 1] = b;
			edge_destination[i + 1] = a;
			weights[i + 1] = 1;

			i += 2;
		}

		total_weight = n_links;

		printf("Graph Host created! \n");

	}
};

#endif
