#ifndef OPTIMIZATION_H     
#define OPTIMIZATION_H

#include "community.h"
#include "constants.h"
#include "operators.h"
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>



class OptimizationPhase {
private:
	thrust::device_vector<bool> neighboorhood_change;

	thrust::device_vector<unsigned int> key_node_source;
	thrust::device_vector<unsigned int> key_community_dest;
	thrust::device_vector<float> values_weight;

	thrust::device_vector<unsigned int> reduced_key_source;
	thrust::device_vector<unsigned int> reduced_key_dest;
	thrust::device_vector<float> reduced_value;

	Community& community;
	int execution_number = 0;


	OptimizationPhase(Community& c) :
		neighboorhood_change(thrust::device_vector<bool>(c.graph.n_nodes, true)),
		community(c){
	
		key_node_source = thrust::device_vector<unsigned int>(community.graph.edge_source.size());
		key_community_dest = thrust::device_vector<unsigned int>(community.graph.edge_source.size());
		values_weight = thrust::device_vector<float>(community.graph.edge_source);

		reduced_key_source = thrust::device_vector<unsigned int>(community.graph.edge_source.size());
		reduced_key_dest = thrust::device_vector<unsigned int>(community.graph.edge_source.size());
		reduced_value = thrust::device_vector<float>(community.graph.edge_source.size());

	}

	void optimize();


	void internal_run() {
		float old_modularity = 0;
		float delta = 0;

		do {
			execution_number++;
			old_modularity = community.modularity;
			optimize();
			community.compute_modularity();
			delta = community.modularity - old_modularity;
		
		#if PRINT_DEBUG_LOG
			printf("MODULARITY = %10f\n", community.modularity);
			printf("Delta Modularity iteration %d: %10f \n", execution_number, delta);
		#endif 

		} while ((execution_number <= EARLY_STOP_LIMIT) && (delta > MODULARITY_CONVERGED_THRESHOLD));

	}

public:
	static void run(Community& c, bool fastLocalMove) {
		OptimizationPhase optimizer = OptimizationPhase(c);

		#if PRINT_PERFORMANCE_LOG
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
		#endif
		
		optimizer.internal_run();

		#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "Total Optimization Time : " << milliseconds << "ms" << std::endl;		
		#endif
	}
};

#endif