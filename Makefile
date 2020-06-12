
all: phaseOptimizationUpdate.o phaseOptimizationHash.o phaseOptimizationSort.o
	nvcc -O3 main.cu phaseOptimizationUpdate.o phaseOptimizationHash.o phaseOptimizationSort.o graph.o -o app

phaseOptimizationUpdate.o: graph.o
	nvcc -dc -O3 phaseOptimizationUpdate.cu 

phaseOptimizationHash.o: graph.o
	nvcc -dc -O3 phaseOptimizationHash.cu 

phaseOptimizationSort.o: graph.o
	nvcc -dc -O3 phaseOptimizationSort.cu 

graph.o:
	nvcc -dc -O3 graph.cu

clean:
	rm -f *.o app

