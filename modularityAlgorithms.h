#ifndef MOD_ALG_H     
#define MOD_ALG_H

#include "phaseOptimization.cuh"
#include "phaseRefine.cuh"
#include "phaseAggregation.cuh"

struct ModularityAlgorithms {
	static Community Louvain(Graph& graph, bool fastLocalMoveEnable) {
		return modularity_routine(graph, false, fastLocalMoveEnable);;
	};

	static Community Laiden(Graph& graph) {
		return modularity_routine(graph, true, true);
	}

private:
	static Community modularity_routine(Graph graph, bool isLaiden, bool fastLocalMove) {
		Community C = Community(graph);
		double old_modularity = 0;
		do {
			old_modularity = C.modularity;
			OptimizationPhase::run(C, fastLocalMove || isLaiden);
			if (isLaiden) {
				RefinePhase::run();
			}
			AggregationPhase::run(C);

		} while (C.modularity - old_modularity >= MODULARITY_CONVERGED_THRESHOLD);

		return C;
	}
};


#endif