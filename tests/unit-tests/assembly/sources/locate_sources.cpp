
#include "../test_fixture/test_fixture.hpp"
#include "algorithms/locate_point.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include "test_macros.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>

TEST_F(ASSEMBLY, locate_sources) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    const auto source_solution = Test.solutions.source;
    auto sources = std::get<2>(parameters);
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly =
        std::get<5>(parameters);
    std::cout << "Running test: " << Test.name << "\n" << std::endl;
    EXPECT_REAL_EQ(sources.size(), 1);
    EXPECT_REAL_EQ(sources[0]->get_xi(), source_solution.xi);
    EXPECT_REAL_EQ(sources[0]->get_gamma(), source_solution.gamma);
    EXPECT_EQ(sources[0]->get_element_index(), source_solution.ispec);
    EXPECT_TRUE(sources[0]->get_medium_tag() == source_solution.medium_tag)
        << "Expected medium tag: "
        << specfem::element::to_string(source_solution.medium_tag)
        << ", but got: "
        << specfem::element::to_string(sources[0]->get_medium_tag());
    std::cout << "Done test: " << Test.name << "\n" << std::endl;
  }
};
