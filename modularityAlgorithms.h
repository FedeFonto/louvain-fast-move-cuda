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
	static Community modularity_routine(GraphHost graph, bool isLaiden, bool fastLocalMove) {
		Community C = Community(graph);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		double old_modularity = C.modularity;
		OptimizationPhase::run(C, fastLocalMove || isLaiden);
		/*if (isLaiden) {
			RefinePhase::run();
		}*/

		while( C.modularity - old_modularity > MODULARITY_CONVERGED_THRESHOLD ) {
			AggregationPhase::run(C);
			old_modularity = C.modularity;
			OptimizationPhase::run(C, fastLocalMove || isLaiden);
			/*if (isLaiden) {
				RefinePhase::run();
			}*/
		};

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "\nFinal Modularity: " << C.modularity << "\nTotal Execution Time: " << milliseconds << "ms" << std::endl;

		return C;
	}
};


#endif