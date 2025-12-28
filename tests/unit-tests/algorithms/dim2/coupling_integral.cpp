
#include "Kokkos_Core.hpp"
#include "execution/chunked_intersection_iterator.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "utilities/include/fixture/assembly.hpp"

#include "algorithms/integrate/coupling_integral1d.hpp"

#include <sstream>

/**
 * @brief specfem::chunk_edge::intersection_factor monkey patch to remove
 * scratch view
 *
 * @tparam InterfaceTag
 * @tparam BoundaryTag
 * @tparam NumberElements
 * @tparam NQuadIntersection
 * @tparam MemorySpace
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection, typename MemorySpace>
struct intersection_factor_patch
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::intersection_factor,
          specfem::dimension::type::dim2, false> {

public:
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
  using IntersectionFactorViewType =
      Kokkos::View<type_real[NumberElements][NQuadIntersection], MemorySpace>;

private:
  IntersectionFactorViewType data_;

public:
  KOKKOS_INLINE_FUNCTION
  intersection_factor_patch() = default;

  KOKKOS_INLINE_FUNCTION intersection_factor_patch(const std::string &name)
      : data_(name) {}

  // needed for accessor
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

template <specfem::interface::interface_tag interface_tag,
          specfem::element::boundary_tag boundary_tag>
void verify_interfaces(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {
  constexpr int nquad_intersection = 5;
  constexpr int ngll = 5;
  constexpr type_real tol = 1e-6;

  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto connection_tag = specfem::connections::type::nonconforming;
  constexpr bool using_simd = false;
  constexpr auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  constexpr int ncomp_self =
      specfem::element::attributes<dimension_tag, self_medium>::components;

  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);

  const int &n_edges = self_edges.n_edges;
  if (n_edges == 0) {
    return;
  }

  // ==============
  // log initialize
  // ==============

  std::ostringstream oss;
  oss << "Interface execution (" << n_edges << " edges):\n";
  oss << "  - interface: " << specfem::interface::to_string(interface_tag)
      << " (self_medium = " << specfem::element::to_string(self_medium)
      << ")\n";
  oss << "  - boundary: " << specfem::element::to_string(boundary_tag) << "\n";

  constexpr int num_trials_per_edge = nquad_intersection;
  oss << "Each edge tested with " << num_trials_per_edge
      << " intersection fields (all " << nquad_intersection
      << " intersection basis functions"
      // <<", then constant 1"
      << ")\n";
  oss << "Tolerance: " << std::scientific << tol << std::fixed << "\n";

  // ==========
  // data types
  // ==========
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  using parallel_config =
      specfem::parallel_configuration::default_chunk_edge_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>;

  using IntersectionFieldViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, parallel_config::chunk_size, nquad_intersection,
      ncomp_self, using_simd, memory_space,
      Kokkos::MemoryTraits<
          Kokkos::RandomAccess> >; // needed VectorChunkEdgeViewType since
                                   // IntersectionFieldViewType::using_simd used
                                   // in algorithms::coupling_integral.

  using IntersectionFactorViewType =
      intersection_factor_patch<interface_tag, boundary_tag,
                                parallel_config::chunk_size, nquad_intersection,
                                memory_space>;

  using ExpectedIntegralViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, parallel_config::chunk_size, ngll, ncomp_self,
      using_simd, memory_space, Kokkos::MemoryTraits<Kokkos::RandomAccess> >;

  using ResultsView_ipoint_hits =
      Kokkos::View<int * /*edge index*/[ngll] /* ipoint indices */,
                   memory_space, Kokkos::MemoryTraits<Kokkos::Atomic> >;

  // failure gathering (total fail count, get diagnostic data from first n
  // fails)
  constexpr int failbin_num_iedge = 5;
  constexpr int failbin_size = 5;
  using ResultsView_failbins_counts =
      Kokkos::View<int[failbin_num_iedge][ngll] /* ipoint indices */,
                   memory_space, Kokkos::MemoryTraits<Kokkos::Atomic> >;
  using ResultsView_failbins_inds = Kokkos::View<
      int[failbin_num_iedge][ngll] /* ipoint indices*/[failbin_size][5],
      memory_space, Kokkos::MemoryTraits<Kokkos::Atomic> >; // stores (is_set,
                                                            // iedge, ipoint,
                                                            // itrial, icomp)
  using ResultsView_failbins_expect_got = Kokkos::View<
      type_real[failbin_num_iedge][ngll] /* ipoint indices*/[failbin_size][2],
      memory_space>; // stores itrial and icomp

  // ==========
  // init views
  // ==========
  IntersectionFieldViewType intersection_field("intersection field");
  IntersectionFactorViewType intersection_factor("intersection factor");
  ExpectedIntegralViewType expected_integral("expected integral");

  ResultsView_ipoint_hits result_ipoint_hits("result: ipoint hits", n_edges);
  Kokkos::deep_copy(result_ipoint_hits, 0);

  ResultsView_failbins_counts failures_counts("result fails: counts");
  ResultsView_failbins_inds failures_inds("result fails: indices");
  ResultsView_failbins_expect_got failures_expect_got(
      "result fails: expect / got");
  Kokkos::deep_copy(failures_counts, 0);
  Kokkos::deep_copy(failures_inds, 0);
  // ============
  // start kernel
  // ============

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges);
  specfem::execution::for_each_level(
      "algorithms/dim2/coupling_integral.cpp::coupling_integral", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const auto &self_chunk_iterator_index = chunk_index.get_self_index();
        const auto self_chunk_index = self_chunk_iterator_index.get_index();

        const auto &team = self_chunk_index.get_policy_index();
        const int index_offset =
            team.league_rank() * parallel_config::chunk_size;
        const int &current_nelem = self_chunk_index.nedges();

        // ========================
        // ========================
        // BEGIN RELEVANT TEST CODE

        // ====================================
        // fill parameters and expected results
        // ====================================
        specfem::assembly::load_on_device(self_chunk_index,
                                          assembly.nonconforming_interfaces,
                                          intersection_factor);

        for (int itrial = 0; itrial < num_trials_per_edge; ++itrial) {
          // set intersection field and expected integrals

          // start with zeroing out everything
          specfem::execution::for_each_level(
              specfem::execution::TeamThreadMDRangeIterator(
                  team, current_nelem, nquad_intersection, ncomp_self),
              [&](const auto &index) {
                intersection_field(index(0), index(1), index(2)) = 0;
              });
          specfem::execution::for_each_level(
              specfem::execution::TeamThreadMDRangeIterator(team, current_nelem,
                                                            ngll, ncomp_self),
              [&](const auto &index) {
                expected_integral(index(0), index(1), index(2)) = 0;
              });

          if (itrial < nquad_intersection) { // case: shape function
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, current_nelem),
                [&](const auto &ielem) {
                  const int ielem_global = index_offset + ielem;
                  // round robin for nonzero component
                  const int nonzero_component =
                      (ielem_global * num_trials_per_edge + itrial) %
                      ncomp_self;

                  intersection_field(ielem, itrial, nonzero_component) =
                      1; // delta(itrial, iintersection)

                  for (int ipoint = 0; ipoint < ngll; ++ipoint) {
                    // should be kronecker on iintersection (itrial), in the
                    // integral sum

                    // only iedge and ipoint is needed for transfer function
                    // retrieval
                    specfem::point::edge_index<dimension_tag> self_index(
                        0, ielem_global, ipoint, 0, 0,
                        specfem::mesh_entity::dim2::type::top);
                    specfem::point::transfer_function_self<
                        nquad_intersection, dimension_tag, interface_tag,
                        boundary_tag>
                        self_transfer;
                    specfem::assembly::load_on_device(
                        self_index, assembly.nonconforming_interfaces,
                        self_transfer);
                    expected_integral(ielem, ipoint, nonzero_component) =
                        intersection_factor(ielem, itrial) *
                        self_transfer(itrial);
                  }
                });
          }

          // =============================
          // compute and compare solutions
          // =============================

          specfem::algorithms::coupling_integral(
              assembly, self_chunk_index, intersection_field,
              intersection_factor,
              [&](const auto &self_index, auto &self_field) {
                // later: when handling interiors, ipoint indices must change
                const int &result_edge_index = self_index.iedge;
                const int &result_point_index = self_index.ipoint;

                ++result_ipoint_hits(result_edge_index, result_point_index);

                const int &failbin_iedge =
                    result_edge_index % failbin_num_iedge;

                for (int icomp = 0; icomp < ncomp_self; icomp++) {
                  const type_real &expect =
                      expected_integral(result_edge_index - index_offset,
                                        self_index.ipoint, icomp);
                  const type_real &got = self_field(icomp);
                  if (std::abs(got - expect) > tol) {
                    const int failind = failures_counts(
                        result_edge_index,
                        result_point_index)++; // retrieve, then increment (same
                                               // op for concurrency)

                    // additional info if needed
                    if (failind < failbin_size) {
                      // failures_inds stores (is_set, iedge, ipoint, itrial,
                      // icomp)
                      ++failures_inds(failbin_iedge, result_point_index,
                                      failind, 0);
                      failures_inds(failbin_iedge, result_point_index, failind,
                                    1) = result_edge_index;
                      failures_inds(failbin_iedge, result_point_index, failind,
                                    2) = result_point_index;
                      failures_inds(failbin_iedge, result_point_index, failind,
                                    3) = itrial;
                      failures_inds(failbin_iedge, result_point_index, failind,
                                    4) = icomp;

                      failures_expect_got(failbin_iedge, result_point_index,
                                          failind, 0) = expect;
                      failures_expect_got(failbin_iedge, result_point_index,
                                          failind, 1) = got;
                    }
                  }
                }
              });
        }

        //  END RELEVANT TEST CODE
        // ========================
        // ========================
      });

  // ==============
  // verify results
  // ==============

  // ensuring everything got hit.
  typename ResultsView_ipoint_hits::HostMirror h_result_ipoint_hits =
      Kokkos::create_mirror_view(result_ipoint_hits);
  Kokkos::deep_copy(h_result_ipoint_hits, result_ipoint_hits);

  bool is_fail = false;

  oss << "Counting number of times algorithms::coupling_integral called-back "
         "each edge point (expects "
      << num_trials_per_edge << ")...\n";

  for (int iedge = 0; iedge < n_edges; iedge++) {
    for (int ipoint = 0; ipoint < ngll; ipoint++) {
      const int &nhits = h_result_ipoint_hits(iedge, ipoint);
      if (nhits != num_trials_per_edge) {
        is_fail = true;
        oss << "  [iedge = " << iedge << ", ipoint = " << ipoint << "]: got "
            << nhits << "\n";
      }
    }
  }

  oss << "Done!\n";

  // retrieve failures from bins (if they exist)

  oss << "Gathering failures...\n";

  typename ResultsView_failbins_counts::HostMirror h_failures_counts =
      Kokkos::create_mirror_view(failures_counts);
  typename ResultsView_failbins_inds::HostMirror h_failures_inds =
      Kokkos::create_mirror_view(failures_inds);
  typename ResultsView_failbins_expect_got::HostMirror h_failures_expect_got =
      Kokkos::create_mirror_view(failures_expect_got);
  Kokkos::deep_copy(h_failures_counts, failures_counts);
  Kokkos::deep_copy(h_failures_inds, failures_inds);
  Kokkos::deep_copy(h_failures_expect_got, failures_expect_got);

  for (int iedge_bin = 0; iedge_bin < failbin_num_iedge; ++iedge_bin) {
    for (int ipoint = 0; ipoint < ngll; ++ipoint) {
      const int &nfails = h_failures_counts(iedge_bin, ipoint);
      if (nfails > 0) {
        is_fail = true;
        const int nfail_print = std::min(failbin_size, nfails);
        oss << "  Bin (" << iedge_bin << ", " << ipoint << ") has collected "
            << nfails << " failures. Listing " << nfail_print << ":\n";

        for (int ifail = 0; ifail < nfail_print; ++ifail) {
          // stores (is_set, iedge, ipoint, itrial, icomp)
          oss << "  - " << ifail << ": ";
          if (h_failures_inds(iedge_bin, ipoint, ifail, 0) != 1) {
            oss << "[Missing or corrupted in collection] gathered "
                << h_failures_inds(iedge_bin, ipoint, ifail, 0)
                << "in index. Should be 1.\n";
          } else {
            const int &f_iedge = h_failures_inds(iedge_bin, ipoint, ifail, 1);
            const int &f_ipoint = h_failures_inds(iedge_bin, ipoint, ifail, 2);
            const int &f_itrial = h_failures_inds(iedge_bin, ipoint, ifail, 3);
            const int &f_icomp = h_failures_inds(iedge_bin, ipoint, ifail, 4);
            const type_real &expected =
                h_failures_expect_got(iedge_bin, ipoint, ifail, 0);
            const type_real &got =
                h_failures_expect_got(iedge_bin, ipoint, ifail, 1);
            oss << "\n      [iedge = " << f_iedge << ", ipoint = " << f_ipoint
                << ", trial = " << f_itrial << ", component = " << f_icomp
                << "]\n"
                << "      expected: " << expected << "\n"
                << "           got: " << got;
            if (std::abs(expected) > 1e-4) {
              oss << " (rel: " << std::showpos << std::scientific
                  << (got - expected) / std::abs(expected) << std::fixed
                  << std::noshowpos << ")";
            }

            oss << "\n";
          }
        }
      }
    }
  }

  oss << "Done!\n";

  if (is_fail) {
    ADD_FAILURE() << oss.str();
  }
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
 * @brief struct for naming the typed test suite.
 * http://google.github.io/googletest/reference/testing.html#TYPED_TEST_SUITE
 *
 */
struct MeshedCouplingIntegralTest2DNames {
  template <typename TestingTypes> static std::string GetName(int) {
    return specfem::test_fixture::impl::name<
        std::tuple_element_t<0, TestingTypes> >::get();
  }
};

/**
 * @brief Test fixture for 2D transfer function algorithms.
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes>
struct MeshedCouplingIntegralTest2D : public ::testing::Test {
  using AssemblyInitializer = std::tuple_element_t<0, TestingTypes>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  MeshedCouplingIntegralTest2D() {}

  specfem::test_fixture::Assembly2D<AssemblyInitializer> assembly_fixture;

  static void print_description() {
    std::ostringstream oss;
    oss << "====================================================\n";
    oss << "-=-=- Test: "
        << MeshedCouplingIntegralTest2DNames::GetName<TestingTypes>(0)
        << " -=-=-\n";
    oss << "  Assembly:\n";
    oss << specfem::test_fixture::impl::description<AssemblyInitializer>::get(
        4);
    oss << "\n====================================================\n";
    SPECFEMEnvironment::get_mpi()->cout(oss.str());
  }
};

using namespace specfem::test_fixture;

/** Test type combinations for parameterized testing */
using MeshedCouplingIntegralTestTypes2D =
    ::testing::Types<std::tuple<AssemblyInitializer2D::FromMesh<
        MeshInitializer2D::ThreeElementNonconforming> > >;

TYPED_TEST_SUITE(MeshedCouplingIntegralTest2D,
                 MeshedCouplingIntegralTestTypes2D,
                 MeshedCouplingIntegralTest2DNames);

TYPED_TEST(MeshedCouplingIntegralTest2D, FullMeshOnKroneckeredFields) {
  this->print_description();
  const auto assembly = this->assembly_fixture.assembly_instance();
  execute(*assembly);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
