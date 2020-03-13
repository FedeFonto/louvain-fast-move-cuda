#ifndef OPTIMIZATION_H     
#define OPTIMIZATION_H

#include "community.h"
#include "constants.h"
#include "operators.h"
#include <thrust\count.h>
#include <thrust\remove.h>
#include <thrust\sort.h>
#include <thrust\reduce.h>
#include <thrust\extrema.h>



class OptimizationPhase {
private:
	thrust::device_vector<int> buckets;
	thrust::device_vector<bool> neighboorhood_change;

	thrust::device_vector<int> old_communities;
	thrust::device_vector<float> old_communities_weight;

	float old_modularity = 0;


	Community& community;


	OptimizationPhase(Community& c) :
		buckets(thrust::device_vector<int>(c.graph.n_nodes)),
		neighboorhood_change(thrust::device_vector<bool>(c.graph.n_nodes, true)),
		old_communities(thrust::device_vector<int>(c.graph.n_nodes)),
		old_communities_weight(thrust::device_vector<int>(c.graph.n_nodes)),
		community(c)
	{
		auto ran = thrust::device_vector<int>(BUCKETS_RANGE);
		thrust::transform(
			c.graph.n_of_neighboor.begin(),
			c.graph.n_of_neighboor.end(),
			buckets.begin(),
			InWitchBucket(thrust::raw_pointer_cast(ran.data()), ran.size())
		);

		// Node Stats: 
		/*
		int result0 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(0));
		int result1 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(1));
		int result2 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(2));
		int result3 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(3));
		int result4 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(4));
		int result5 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(5));
		int result6 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(6));
		int result7 = thrust::count_if(thrust::device, buckets.begin(), buckets.end(), IsEquals(7));
		*/

	}


	/*
	void calculate_bucket_one_time() {
		for (int i = 0; i < offset.size(); i++) {
			offset[i] = thrust::count_if(buckets.begin(), buckets.end(), IsEquals(i));
		}

		for (int i = 1; i < offset.size(); i++) {
			offset[i] += offset[i - 1];
		}

		thrust::copy_if(buckets.begin(), buckets.end(), indices.begin(), selected_nodes.begin(), IsEquals(0));
		for (int i = 1; i < offset.size(); i++) {
			thrust::copy_if(buckets.begin(), buckets.end(), indices.begin(), selected_nodes.begin() + offset[i - 1], IsEquals(i));
		}
	}*/

	void optimize();

	void backup() {
		old_modularity = community.modularity;
		thrust::copy(community.communities.begin(), community.communities.end(), old_communities.begin());
		thrust::copy(community.communities_weight.begin(), community.communities_weight.end(), old_communities_weight.begin());
	}

	void rollback() {
		community.modularity = old_modularity;
		thrust::copy(old_communities.begin(), old_communities.end(), community.communities.begin());
		thrust::copy(old_communities_weight.begin(), old_communities_weight.end(),community.communities_weight.begin());
	}


	void internal_run() {
		int execution_number = 0;
		float new_modularity = 0;
		do {
			backup();
			optimize();
			community.compute_modularity();
			new_modularity = community.modularity;
			if (new_modularity - old_modularity <= 0) {
				rollback();
			}
			execution_number++;
		
		#if PRINT_DEBUG_LOG
			printf("Delta Modularity iteration %d: %10f \n", execution_number, new_modularity - old_modularity);
		#endif 

		} while ((execution_number <= EARLY_STOP_LIMIT) && (new_modularity - old_modularity > MODULARITY_CONVERGED_THRESHOLD));

		#if PRINT_DEBUG_LOG
			printf("MODULARITY = %10f\n", community.modularity);
		#endif 
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
			std::cout << "Optimization Time : " << milliseconds << "ms" << std::endl;		
		#endif
	}
};

#endif