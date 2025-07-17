
#include "../test_fixture/test_fixture.hpp"
#include "algorithms/locate_point.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include "test_macros.hpp"
#include "utilities/utilities.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>

TEST_F(ASSEMBLY, locate_sources) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    const auto source_solution = Test.solutions.source;
    auto sources = std::get<2>(parameters);
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly =
        std::get<5>(parameters);

    // Use SCOPED_TRACE to make each iteration identifiable in test output
    SCOPED_TRACE(Test.name);

    // Check the number of sources
    EXPECT_REAL_EQ(sources.size(), 1);

    // Check xi
    EXPECT_TRUE(specfem::utilities::almost_equal(sources[0]->get_xi(),
                                                 source_solution.xi, 1e-4))
        << ExpectedGot(source_solution.xi, sources[0]->get_xi());

    // Check gamma
    EXPECT_TRUE(specfem::utilities::almost_equal(sources[0]->get_gamma(),
                                                 source_solution.gamma, 1e-4))
        << ExpectedGot(source_solution.gamma, sources[0]->get_gamma());

    // Check ispec
    EXPECT_EQ(sources[0]->get_element_index(), source_solution.ispec);

    // Check medium tag/medium that the source is located in
    EXPECT_TRUE(sources[0]->get_medium_tag() == source_solution.medium_tag)
        << "Expected medium tag: "
        << specfem::element::to_string(source_solution.medium_tag)
        << ", but got: "
        << specfem::element::to_string(sources[0]->get_medium_tag());
  }
};
