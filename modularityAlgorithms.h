#ifndef MOD_ALG_H     
#define MOD_ALG_H

#include "phaseOptimization.cuh"
#include "phaseAggregation.cuh"

struct ModularityAlgorithms {
	static Community Louvain(GraphHost& graph, MODE fastLocalMoveEnable) {
		return modularity_routine(graph, fastLocalMoveEnable);
	};


private:
	static Community modularity_routine(GraphHost& graph, MODE mode) {
		printf("Creating Device Graph..\n");
		Community C = Community(graph);

		auto stats = thrust::device_vector<float>(6);
		stats[0] = graph.n_nodes;
		stats[1] = graph.n_links;

		printf("Start of Optimization Algorithm\n");
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		int iteration = 0;
		float old_modularity = 0;
		bool continue_optimization = false;
		do {
#if PRINT_PERFORMANCE_LOG
			cudaEvent_t r_start, r_stop, a_stop;
			cudaEventCreate(&r_start);
			cudaEventCreate(&r_stop);
			cudaEventCreate(&a_stop);
			cudaEventRecord(r_start);
#endif
			old_modularity = C.modularity;
			OptimizationPhase::run(C, mode);

#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(r_stop);
			cudaEventSynchronize(r_stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, r_start, r_stop);

			if (iteration == 0) {
				stats[2] = milliseconds;
			}
#endif
			continue_optimization = (C.modularity - old_modularity) > MODULARITY_CONVERGED_THRESHOLD;
			AggregationPhase::run(C, continue_optimization, mode);

#if PRINT_PERFORMANCE_LOG
			cudaEventRecord(a_stop);
			cudaEventSynchronize(a_stop);
			milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, r_stop, a_stop);

			if (iteration == 0) {
				stats[3] = milliseconds;
			}
#endif
			iteration++;

		} while (continue_optimization);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		stats[4] = milliseconds;
		stats[5] = C.modularity;
		std::cout << "\nFinal Modularity: " << C.modularity << "\nTotal Execution Time: " << milliseconds << "ms" << std::endl;
		for (int i = 0; i < 6; i++)
			std::cout <<stats[i] << ",";
		std::cout<< std::endl;
		return C;
	}
};



#endif