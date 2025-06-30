#include "../test_fixture/test_fixture.hpp"
#include "specfem/compute.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>

void test_check_jacobian(const specfem::compute::assembly &assembly) {

  const auto nspec = assembly.mesh.nspec;

  const specfem::point::index<specfem::dimension::type::dim2, false> index(
      static_cast<int>(nspec / 2), 2, 2);

  const specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                        false>
      jacobian_matrix(0, 0, 0, 0, -0.5);

  specfem::compute::store_on_host(index, assembly.jacobian_matrix,
                                  jacobian_matrix);

  assembly.check_small_jacobian();
}

TEST_F(ASSEMBLY, CheckJacobian) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    specfem::compute::assembly assembly = std::get<5>(parameters);

    bool exception_thrown = false;
    try {
      test_check_jacobian(assembly);
    } catch (const std::exception &e) {
      exception_thrown = true;
    }
    EXPECT_TRUE(exception_thrown)
        << "Expected an exception to be thrown for test in check_jacobian.cpp"
        << " but none was thrown for test: " << Test.name;
  }
}
