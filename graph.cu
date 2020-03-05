#include "graph.cuh"

#include <iterator>
#include <fstream>
#include <map>
#include <vector>

using namespace std;


Graph::Graph(std::string name, bool weighted ) {
	n_links = 0;

	ifstream f;
	f.open(name);

	int a, b;
	auto node_map = map<int, vector<int>>();

	printf("Start Parsing...\n");
	if (weighted) {
		//todo parsing 
	} else {
		while (f >> a >> b) {
			if (node_map.find(a) == node_map.end()) {
				auto v = vector<int>();
				v.push_back(b);
				node_map.insert(pair<int, vector<int>>(a, v));
			}
			else {
				node_map.at(a).push_back(b);
			}

			if (node_map.find(b) == node_map.end()) {
				auto v = vector<int>();
				v.push_back(a);
				node_map.insert(pair<int, vector<int>>(b, v));
			}
			else {
				node_map.at(b).push_back(a);
			}
			n_links++;
		}

		total_weight = n_links;
	}

	n_nodes = node_map.size();

	degrees = thrust::device_vector<unsigned long>(n_nodes + 1);
	n_of_neighboor = thrust::device_vector<int>(n_nodes);
	tot_weight_per_nodes = thrust::device_vector<float>(n_nodes);

	edge_source = thrust::device_vector<int>(2 * n_links);
	edge_destination = thrust::device_vector<int>(2 * n_links);
	weights = thrust::device_vector<float>(2 * n_links);

	printf("Creating Graph Object...\n");

	unsigned int edge_count = 0;
	for (unsigned int nodes = 0; nodes < n_nodes; nodes++) {
		auto element = node_map[nodes];
		n_of_neighboor[nodes] = element.size();
		tot_weight_per_nodes[nodes] = element.size();
		degrees[nodes] = edge_count;
		for (int edge = 0; edge < element.size(); edge++) {
			edge_source[edge_count + edge] = nodes;
			edge_destination[edge_count + edge] = element[edge];
			if(weighted){
				//todo 
			} else {
				weights[edge_count + edge] = 1;
			}
		}
		edge_count += element.size();
	}


	degrees[n_nodes] = edge_count;
};
