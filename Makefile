
all: phaseOptimization.o
	nvcc -O3 main.cu phaseOptimization.o graph.o -o app

phaseOptimization.o: graph.o
	nvcc -dc -O3 phaseOptimization.cu 

graph.o:
	nvcc -dc -O3 graph.cu

clean:
	rm -f *.o app

