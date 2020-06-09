
#ifndef GRAPH_CU   
#define GRAPH_CU

#include "graphHost.cuh"
#include "operators.h"

#include <iterator>
#include <fstream>
#include <set>
#include <vector>
#include <map>


using namespace std;


GraphHost::GraphHost(std::string name, bool weighted, int skip_line) {
	ifstream f;
	f.open(name);
	
	string ignore;
	for (int i = 0; i < skip_line;  i++) {
		getline(f, ignore);
	}

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
			edge_set.insert(make_pair(a,b));

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
		for (auto it = edge_set.begin(); it != edge_set.end(); ++it) {
			edge_source[i] = it->first;
			edge_destination[i] = it->second;
			weights[i] = 1;

			edge_source[i + 1] = it->second;
			edge_destination[i + 1] = it->first;
			weights[i + 1] = 1;

			i += 2;
		}

		total_weight = n_links;
		min_value  = *node_set.begin();

		cout<< "Graph Host created! Name: " << name << ", nodes: " << n_nodes << ", edges: " << n_links << endl;

	}
};

#endif
