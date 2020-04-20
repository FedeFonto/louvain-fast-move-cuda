#include "modularityAlgorithms.h"

int main()
{

    GraphHost g = GraphHost::GraphHost("soc-LiveJournal.txt", false);

    auto C = ModularityAlgorithms::Laiden(g);
    do
    {
        std::cout << '\n' << "Press enter to continue...";
    } while (std::cin.get() != '\n');
    return 0;
}
