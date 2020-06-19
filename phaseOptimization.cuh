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
#include "hashmap.cuh"


class OptimizationPhase {
private:
	thrust::device_vector<bool> neighboorhood_change;
	thrust::device_vector<bool> its_changed;

	thrust::device_vector<unsigned int> key_node_source;
	thrust::device_vector<unsigned int> key_community_dest;
	thrust::device_vector<float> values_weight;

	thrust::device_vector<unsigned int> reduced_key_source;
	thrust::device_vector<unsigned int> reduced_key_dest;
	thrust::device_vector<float> reduced_value;

	thrust::device_vector<unsigned int> final_node;
	thrust::device_vector<unsigned int> final_community;
	thrust::device_vector<float> final_value;

	thrust::device_vector<unsigned long long int> final_pair;

	unsigned int nodes_considered = 0;

	Community& community;
	int execution_number = 0;

	const static int adaptive = 4;

	MODE mode;
	HashMap* hashmap;

#if PRINT_PERFORMANCE_LOG
	std::vector<float> performance = std::vector<float>(10,0);
#endif
#if PRINT_DEBUG_LOG
	std::vector<float> modularity = std::vector<float>(10, 0);
	std::vector<float> delta_mod = std::vector<float>(10, 0);
	std::vector<int> n_comm = std::vector<int>(10, 0);
#endif
	

	OptimizationPhase(Community& c, MODE m) :
		mode(m),
		neighboorhood_change(thrust::device_vector<bool>(c.graph.n_nodes, true)),
		community(c){
	
		its_changed = thrust::device_vector<bool>(community.graph.n_nodes, false);

		key_node_source = thrust::device_vector<unsigned int>(STEP_ROUND);
		key_community_dest = thrust::device_vector<unsigned int>(STEP_ROUND);
		values_weight = thrust::device_vector<float>(STEP_ROUND);

		reduced_key_source = thrust::device_vector<unsigned int>(STEP_ROUND);
		reduced_key_dest = thrust::device_vector<unsigned int>(STEP_ROUND);
		reduced_value = thrust::device_vector<float>(STEP_ROUND);
		
		final_node = thrust::device_vector<unsigned int>(c.graph.n_nodes, 0);
		final_community = thrust::device_vector<unsigned int>(c.graph.n_nodes, 0);
		final_value = thrust::device_vector<float>(c.graph.n_nodes, -1);
	}


	~OptimizationPhase() {
		if (((mode == HASH || mode == ADAPTIVE_MEMORY ) && execution_number > 1) || (mode == ADAPTIVE_SPEED && execution_number > (adaptive +1))) {
			delete hashmap;
		}
	}

	void swap_mode() {
		key_node_source.clear();
		key_node_source.shrink_to_fit();

		key_community_dest.clear();
		key_community_dest.shrink_to_fit();

		values_weight.clear();
		values_weight.shrink_to_fit();

		reduced_key_source.clear();
		reduced_key_source.shrink_to_fit();

		reduced_key_dest.clear();
		reduced_key_dest.shrink_to_fit();

		reduced_value.clear();
		reduced_value.shrink_to_fit();

		final_node.clear();
		final_node.shrink_to_fit();

		final_community.clear();
		final_community.shrink_to_fit();

		final_value.clear();
		final_value.shrink_to_fit();

		final_pair = thrust::device_vector<unsigned long long int>(community.graph.n_nodes, 0);
		hashmap = new HashMap(BUCKETS_SIZE);
	}
	

	void optimize_hash();
	void optimize_sort();
	void optimize_fast();

	void fast_move_update(bool value);

	void optimize() {
#if PRINT_PERFORMANCE_LOG
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
#endif
		nodes_considered = 0;

		if ((mode == ADAPTIVE_SPEED && execution_number == adaptive) || ((mode == HASH || mode == ADAPTIVE_MEMORY ) && execution_number == 1)) {
			swap_mode();
		}

		if (execution_number == 0) {
			thrust::fill(final_value.begin(), final_value.end(), -1);
			optimize_fast();
			fast_move_update(false);
		}
		else {
			const bool useHash = mode == HASH || mode == ADAPTIVE_MEMORY || (mode == ADAPTIVE_SPEED && execution_number > adaptive);
			if (useHash) {
				thrust::fill(final_pair.begin(), final_pair.end(), 0);
				optimize_hash();
			}
			//if mode == SORT or (mode == ADAPTIVE_SPEED and execution_number <= 3)
			else {
				thrust::fill(final_value.begin(), final_value.end(), -1);
				optimize_sort();
			}
			fast_move_update(useHash);
		}

#if PRINT_PERFORMANCE_LOG
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		if(execution_number < 10)
			performance[execution_number] = milliseconds;
#endif
	};


	void internal_run() {
		double old_modularity = 0;
		double delta = 0;

#if  PRINT_PERFORMANCE_LOG && INCLUDE_SUBPHASE

		std::cout << "Optimization Phase " << MyUtils::mode_name(mode) << std::endl;
#endif 
		old_modularity = community.modularity;
		optimize();
		community.compute_modularity();
		delta = community.modularity - old_modularity;

#if PRINT_DEBUG_LOG
		if (execution_number < 10) {
			modularity[execution_number] = community.modularity;
			delta_mod[execution_number] = delta;
			n_comm[execution_number] = thrust::transform_reduce(
				community.communities_weight.begin(),
				community.communities_weight.end(),
				CountNotZero(),
				0,
				thrust::plus<unsigned>()
			);
		}
#endif 
		execution_number++;


		do {
			old_modularity = community.modularity;
			optimize();
			community.compute_modularity();
			delta = community.modularity - old_modularity;

#if PRINT_DEBUG_LOG
			if (execution_number < 10) {
				modularity[execution_number] = community.modularity;
				delta_mod[execution_number] = delta;
				n_comm[execution_number] = thrust::transform_reduce(
					community.communities_weight.begin(),
					community.communities_weight.end(),
					CountNotZero(),
					0,
					thrust::plus<unsigned>()
				);
			}
#endif 
			execution_number++;


		} while ((execution_number <= EARLY_STOP_LIMIT) && (delta > MODULARITY_CONVERGED_THRESHOLD));

	}

public:
	static void run(Community& c, MODE mode) {
		OptimizationPhase optimizer = OptimizationPhase(c, mode);

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
		std::cout << std::endl << "Iteration Stats: " << std::endl;
		for (int i = 0; i < optimizer.performance.size(); i++ )
			std::cout << optimizer.performance[i] << "," ;
		std::cout << milliseconds << std::endl;
#endif

#if PRINT_DEBUG_LOG
		std::cout << std::endl << "Modularity Stats: " << std::endl;
		for (int i = 0; i < optimizer.modularity.size(); i++)
			std::cout << optimizer.modularity[i] << ",";
		std::cout << std::endl;
		for (int i = 0; i < optimizer.delta_mod.size(); i++)
			std::cout << optimizer.delta_mod[i] << ",";
		std::cout << std::endl;
		for (int i = 0; i < optimizer.n_comm.size(); i++)
			std::cout << optimizer.n_comm[i] << ",";
		std::cout << std::endl;
#endif 
	}
		
};

#endif