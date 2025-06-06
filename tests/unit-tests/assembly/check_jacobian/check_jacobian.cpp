#include "../test_fixture/test_fixture.hpp"
#include "algorithms/locate_point.hpp"
#include "compute/assembly/assembly.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>

void test_check_jacobian(const specfem::compute::assembly &assembly) {

  const auto nspec = assembly.mesh.nspec;

  const specfem::point::index<specfem::dimension::type::dim2, false> index(
      static_cast<int>(nspec / 2), 2, 2);

  const specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                            true, false>
      partial_derivatives(0, 0, 0, 0, -0.5);

  specfem::compute::store_on_host(index, assembly.partial_derivatives,
                                  partial_derivatives);

  assembly.check_small_jacobian();
}

TEST_F(ASSEMBLY, CheckJacobian) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    specfem::compute::assembly assembly = std::get<5>(parameters);

    // Break down the expectation
    bool threw_exception = false;
    try {
        test_check_jacobian(assembly);
    } catch (const std::runtime_error&) {
        threw_exception = true;
    }
    EXPECT_TRUE(threw_exception);
  }
}
