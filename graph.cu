
#ifndef GRAPH_CU   
#define GRAPH_CU

#include "graphHost.cuh"
#include "operators.h"

#include <iterator>
#include <fstream>
#include <set>
#include <vector>


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
	auto ss = 2 * n_links;
	edge_source = thrust::host_vector<int>(ss);
	edge_destination = thrust::host_vector<int>(ss);
	weights = thrust::host_vector<float>(ss);

	int i = 0;
	if (weighted) {
		//todo parsing 
	}
	else {
		for (std::set<pair<int, int>>::iterator it = edge_set.begin(); it != edge_set.end(); ++it) {
			edge_source[i] = it->first;
			edge_destination[i] = it->second;
			weights[i] = 1;

			auto s = i + 1;
			edge_source[s] = it->second;
			edge_destination[s] = it->first;
			weights[s] = 1;

			i += 2;
		}

		total_weight = n_links;
		min_id = *node_set.begin();

		cout<< "Graph Host created! Name: " << name << ", nodes: " << n_nodes << ", edges: " << n_links << endl;

	}
};

#endif
