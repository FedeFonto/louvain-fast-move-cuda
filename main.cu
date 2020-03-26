#include "modularityAlgorithms.h"

int main()
{
    //MyUtils::print_memory_usage();

    GraphHost g = GraphHost::GraphHost("graph-power-law-huge-2.edge", false);

    printf("Start Optimization Algorithm...\n");
    auto C = ModularityAlgorithms::Laiden(g);
    do
    {
        std::cout << '\n' << "Press enter to continue...";
    } while (std::cin.get() != '\n');
    return 0;
}
