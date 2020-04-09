
#ifndef GRAPH_CU   
#define GRAPH_CU

#include "graphHost.cuh"
#include "operators.h"

#include <iterator>
#include <fstream>
#include <set>
#include <vector>


using namespace std;


GraphHost::GraphHost(std::string name, bool weighted) {
	ifstream f;
	f.open(name);

	int a, b;
	auto node_set = set<int>();
	auto edge_set = set<pair<int, int>>();

	printf("Start Parsing...\n");

	if (weighted) {
		//todo parsing 
	}
	else {
		while (f >> a >> b) {
			node_set.insert(a);
			node_set.insert(b);
			if (a > b) {
				edge_set.insert(std::pair<int, int>(b, a));
			}
			else {
				edge_set.insert(std::pair<int, int>(a, b));
			}
		}
	}

	n_links = edge_set.size();
	n_nodes = node_set.size();
	f.clear();

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
		for (std::set<pair<int, int>>::iterator it = edge_set.begin(); it != edge_set.end(); ++it) {
			edge_source[i] = it->first;
			edge_destination[i] = it->second;
			weights[i] = 1;

			edge_source[i + 1] = it->second;
			edge_destination[i + 1] = it->first;
			weights[i + 1] = 1;

			i += 2;
		}

		total_weight = n_links;

		printf("Graph Host created! \n");

	}
};

#endif
