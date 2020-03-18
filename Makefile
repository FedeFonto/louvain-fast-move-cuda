
all: phaseOptimization.o
	nvcc main.cu phaseOptimization.o graph.o -o app

phaseOptimization.o: graph.o
	nvcc -dc phaseOptimization.cu

graph.o:
	nvcc -dc graph.cu

clean:
	rm -f *.o app

