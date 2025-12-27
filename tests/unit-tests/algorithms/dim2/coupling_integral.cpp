
#include "utilities/include/fixture/assembly.hpp"
#include "utilities/include/fixture/assembly/assembly_2d.hpp"
#include "utilities/include/fixture/mesh/mesh.hpp"

void execute(const specfem::assembly::assembly<specfem::dimension::type::dim2>
                 &assembly) {
  assembly.check_jacobian_matrix();
  std::cout << assembly.get_total_number_of_elements();
}

/**
 * @brief Test fixture for 2D transfer function algorithms.
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes>
struct AnalyticalCouplingIntegralTest2D : public ::testing::Test {
  using AssemblyInitializer = std::tuple_element_t<0, TestingTypes>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  AnalyticalCouplingIntegralTest2D() {}

  specfem::test_fixture::Assembly2D<AssemblyInitializer> assembly;
};
using namespace specfem::test_fixture;

/** Test type combinations for parameterized testing */
using AnalyticalCouplingIntegralTestTypes2D =
    ::testing::Types<std::tuple<AssemblyInitializer2D::FromMesh<
        MeshInitializer2D::ThreeElementNonconforming> > >;

TYPED_TEST_SUITE(AnalyticalCouplingIntegralTest2D,
                 AnalyticalCouplingIntegralTestTypes2D);

TYPED_TEST(AnalyticalCouplingIntegralTest2D, ComputeCouplingOnAnalyticalField) {
  execute(this->assembly);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}

/*


# add_executable(
#   coupling_integral_tests
#   algorithms/dim2/coupling_integral.cpp
# )

# target_link_libraries(
#   coupling_integral_tests
#   algorithms
#   enumerations
#   mesh
#   assembly
#   gtest_main
#   Kokkos::kokkos
#   specfem_environment
#   utilities

#   quadrature
#   yaml-cpp
#   parameter_reader
#   compare_arrays
#   timescheme
#   point
#   kokkos_kernels
#   solver
#   periodic_tasks
#   boost
#   -lpthread -lm
# )

*/
