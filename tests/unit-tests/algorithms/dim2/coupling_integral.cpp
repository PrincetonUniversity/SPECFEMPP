
#include "execution/chunked_intersection_iterator.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "utilities/include/fixture/assembly.hpp"
#include <sstream>

template <specfem::interface::interface_tag interface_tag,
          specfem::element::boundary_tag boundary_tag>
void verify_interfaces(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {
  constexpr auto DimensionTag = specfem::dimension::type::dim2;
  constexpr auto connection_tag = specfem::connections::type::nonconforming;
  constexpr auto self_medium =
      specfem::interface::attributes<DimensionTag,
                                     interface_tag>::self_medium();

  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);
  if (self_edges.n_edges == 0)
    return;

  std::ostringstream oss;
  oss << "Interface execution (" << self_edges.n_edges << " edges):\n";
  oss << "  - interface: " << specfem::interface::to_string(interface_tag)
      << " (self_medium = " << specfem::element::to_string(self_medium)
      << ")\n";
  oss << "  - boundary: " << specfem::element::to_string(boundary_tag) << "\n";

  using parallel_config =
      specfem::parallel_configuration::default_chunk_edge_config<
          DimensionTag, Kokkos::DefaultExecutionSpace>;
  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges);

  specfem::execution::for_each_level(
      "specfem::kokkos_kernels::impl::compute_coupling", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const auto &self_chunk_iterator_index = chunk_index.get_self_index();
        const auto self_chunk_index = self_chunk_iterator_index.get_index();

        // specfem::algorithms::coupling_integral(
        //     assembly, self_chunk_index, _TODO_interface_field,
        //     _TODO_integration_factor,
        //     [&](const auto &self_index, auto &self_field) {
        //       // TODO verify here
        //     });
      });

  std::cout << oss.str();
}

void execute(const specfem::assembly::assembly<specfem::dimension::type::dim2>
                 &assembly) {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      { verify_interfaces<_interface_tag_, _boundary_tag_>(assembly); })
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

  specfem::test_fixture::Assembly2D<AssemblyInitializer> assembly_fixture;
};
using namespace specfem::test_fixture;

/** Test type combinations for parameterized testing */
using AnalyticalCouplingIntegralTestTypes2D =
    ::testing::Types<std::tuple<AssemblyInitializer2D::FromMesh<
        MeshInitializer2D::ThreeElementNonconforming> > >;

TYPED_TEST_SUITE(AnalyticalCouplingIntegralTest2D,
                 AnalyticalCouplingIntegralTestTypes2D);

TYPED_TEST(AnalyticalCouplingIntegralTest2D, ComputeCouplingOnAnalyticalField) {
  execute(this->assembly_fixture.assembly_instance());
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
