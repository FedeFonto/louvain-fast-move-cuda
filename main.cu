#include "modularityAlgorithms.h"

int main()
{
    //MyUtils::print_memory_usage();

    Graph g = Graph::Graph("graph-power-law-10000-2-99-1.edge", false);

    printf("Start Optimization Algorithm...\n");
    auto C = ModularityAlgorithms::Laiden(g);
    do
    {
        std::cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
    return 0;
}
