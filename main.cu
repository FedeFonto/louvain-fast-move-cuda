#include "modularityAlgorithms.h"

int main()
{
    //MyUtils::print_memory_usage();

    Graph g = Graph::Graph("graph-power-law-big.edge", false);

    printf("Start Optimization Algorithm...\n");
    auto C = ModularityAlgorithms::Laiden(g);

    return 0;
}
