#ifndef MOD_ALG_H     
#define MOD_ALG_H

#include "phaseOptimization.cuh"
#include "phaseRefine.cuh"
#include "phaseAggregation.cuh"

struct ModularityAlgorithms {
	static Community Louvain(GraphHost& graph, bool fastLocalMoveEnable) {
		return modularity_routine(graph, false, fastLocalMoveEnable);
	};

	static Community Laiden(GraphHost& graph) {
		return modularity_routine(graph, true, true);
	}

private:
	static Community modularity_routine(GraphHost& graph, bool isLaiden, bool fastLocalMove) {
		printf("Creating Device Graph..\n");
		Community C = Community(graph);

		printf("Start of Optimization Algorithm\n");
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		int iteration = 0;
		float old_modularity = 0;
		bool continue_optimization = false;
		do {
			old_modularity = C.modularity;
			OptimizationPhase::run(C, iteration == 0);
			continue_optimization = (C.modularity - old_modularity) > MODULARITY_CONVERGED_THRESHOLD;
			if (isLaiden) {
				RefinePhase::run();
			}
			AggregationPhase::run(C, continue_optimization);
			iteration++;

		} while (continue_optimization);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "\nFinal Modularity: " << C.modularity << "\nTotal Execution Time: " << milliseconds << "ms" << std::endl;

		return C;
	}
};


#endif