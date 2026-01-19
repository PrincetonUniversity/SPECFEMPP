#include "../SPECFEM_Environment.hpp"
#include "medium/compute_coupling.hpp"
#include "specfem/algorithms/transfer.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/parallel_configuration.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>

// We need to simulate a chunk_edge iteration:
template <specfem::dimension::type DimensionTag> class ChunkEdgeIndexSimulator {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndexSimulator(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;
  KokkosIndexType kokkos_index; ///< Kokkos team member for this chunk
};

// base type so that we can use a single value-parameterized test for different
// kernels
struct EdgeToInterfaceParamsBase {
  virtual void run_test(const std::string &testname) const {
    throw std::runtime_error(
        testname +
        std::string(": Called EdgeToInterfaceParamsBase::run_test(). No test "
                    "should be initialized with this type."));
  }
};

/**
 * @brief Evaluates the Lagrange interpolation polynomials at a given point x:
 *     $$L_j(x) = \prod_{k \ne j} \frac{x - \xi_k}{\xi_j - \xi_k}$$
 *
 * @tparam nquad - number of quadrature points, and size of the
 * `quadrature_points` array
 * @param quadrature_points - the array of $\xi_k$
 * @param poly_index - the index of the lagrange polynomial to evaluate
 * @param x - the point to evaluate at
 * @return type_real - the evaluated $L_j(x)$, where $j$ is given by
 * `poly_index`
 */
template <std::size_t nquad>
KOKKOS_FORCEINLINE_FUNCTION type_real
eval_lagrange(const std::array<type_real, nquad> &quadrature_points,
              const int &poly_index, const type_real &x) {
  type_real val = 1;
  for (int i = 0; i < nquad; i++) {
    if (i != poly_index) {
      val *= (x - quadrature_points[i]) /
             (quadrature_points[poly_index] - quadrature_points[i]);
    }
  }
  return val;
}

/**
 * @brief Specialized version of EdgeToInterfaceParamsBase, containing the
 * quadrature points of both the intersection and edge. The virtual method
 * `run_test()` is specialized to the given template parameters
 *
 * @tparam num_edges_ - the chunk size of the chunk_edge views
 * @tparam DimensionTag - dimension of the simulation
 * @tparam InterfaceTag - the interface being modelled. This should not have any
 * bearing on the test, except to set the number of components on either side.
 * @tparam nquad_edge_ - the number of quadrature points on (dimension of) the
 * edge
 * @tparam nquad_intersection_ - the number of quadrature points on (dimension
 * of) the intersections
 */
template <int num_edges_, specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag, int nquad_edge_,
          int nquad_intersection_>
struct EdgeToInterfaceParams : EdgeToInterfaceParamsBase {
  static constexpr specfem::dimension::type dimension_tag = DimensionTag;
  static constexpr specfem::interface::interface_tag interface_tag =
      InterfaceTag;
  static constexpr int num_edges = num_edges_;
  static constexpr int nquad_edge = nquad_edge_;
  static constexpr int nquad_intersection = nquad_intersection_;

  std::array<type_real, nquad_edge_> edge_quadrature_points;
  std::array<type_real, nquad_intersection_> intersection_quadrature_points;

  // takes const array& (lvalue), converts to std::array, which requires data
  // copying. for some reason, I could not get initializer lists to work with
  // std::array.
  EdgeToInterfaceParams(
      const type_real (&edge_quadrature_points)[nquad_edge_],
      const type_real (&intersection_quadrature_points)[nquad_intersection_]) {
    std::copy(std::begin(edge_quadrature_points),
              std::end(edge_quadrature_points),
              std::begin(this->edge_quadrature_points));
    std::copy(std::begin(intersection_quadrature_points),
              std::end(intersection_quadrature_points),
              std::begin(this->intersection_quadrature_points));
  }

  // tolerance for function evaluations (no integration or differentiation, so
  // no dimensions are necessary).
  type_real tolerance() const { return 1e-6; }

  /**
   * @brief Tests the compute_coupling routine (edge to intersection) for the
   * given interface type and quadrature points. The test checks if the
   * polynomials $x^k$ map onto the intersection correctly by transferring the
   * functions
   *
   *     $$x^k = \sum_{j} \xi_j^k L_j(x),~~~ k = 0,\dots, K$$
   *
   * where $K$ is less than `nquad_edge` and `nquad_intersection`, so that both
   * the interpolation and the transfer are exact.
   */
  virtual void run_test(const std::string &testname) const {
    using SelfTransferType = specfem::chunk_edge::transfer_function_self<
        dimension_tag, interface_tag, specfem::element::boundary_tag::none,
        num_edges, nquad_intersection, nquad_edge>;
    using CoupledTransferType = specfem::chunk_edge::transfer_function_coupled<
        dimension_tag, interface_tag, specfem::element::boundary_tag::none,
        num_edges, nquad_intersection, nquad_edge>;

    constexpr auto self_medium =
        specfem::interface::attributes<dimension_tag,
                                       interface_tag>::self_medium();
    constexpr auto coupled_medium =
        specfem::interface::attributes<dimension_tag,
                                       interface_tag>::coupled_medium();

    using SelfDisplacementType =
        specfem::chunk_edge::displacement<num_edges, nquad_edge, dimension_tag,
                                          self_medium, false>;
    using CoupledDisplacementType =
        specfem::chunk_edge::displacement<num_edges, nquad_edge, dimension_tag,
                                          coupled_medium, false>;
    using PointSelfDisplacementType =
        specfem::point::displacement<dimension_tag, self_medium, false>;
    using PointCoupledDisplacementType =
        specfem::point::displacement<dimension_tag, coupled_medium, false>;

    using SelfOnInterfaceType = specfem::datatype::VectorChunkEdgeViewType<
        type_real, specfem::dimension::type::dim2, num_edges,
        nquad_intersection,
        specfem::element::attributes<DimensionTag, self_medium>::components,
        false, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using CoupledOnInterfaceType = specfem::datatype::VectorChunkEdgeViewType<
        type_real, specfem::dimension::type::dim2, num_edges,
        nquad_intersection,
        specfem::element::attributes<DimensionTag, coupled_medium>::components,
        false, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    const type_real tol = tolerance();
    // transfer 1, x, x^2, ..., x^{deg_cap-1}
    constexpr int deg_cap = std::min(nquad_edge, nquad_intersection);

    // iterating this many times guarantees all degrees are run at least once
    const int num_iters = 1 + (deg_cap / num_edges);

    // check for failures through this.
    // We only store the last fail in each thread. (there may be a better way of
    // doing this)

    // we should switch out this tuple for something else. TODO address this in
    // issue #1226
    using CheckContainer = Kokkos::View<
        std::tuple<bool, type_real, type_real, type_real, int, int, int> **,
        Kokkos::DefaultExecutionSpace>;
    CheckContainer self_check_container("self-check container", num_iters,
                                        num_edges);
    CheckContainer coupled_check_container("coupled-check container", num_iters,
                                           num_edges);

    // initialize failure arrays to not-fail state.
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(num_iters, num_edges),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges), [&](const auto &iedge) {
                std::get<0>(self_check_container(team.league_rank(), iedge)) =
                    false;
                std::get<0>(
                    coupled_check_container(team.league_rank(), iedge)) = false;
              });
        });

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(num_iters, num_edges)
            .set_scratch_size(
                0, Kokkos::PerTeam(SelfTransferType::shmem_size() +
                                   CoupledTransferType::shmem_size() +
                                   SelfDisplacementType::shmem_size() +
                                   CoupledDisplacementType::shmem_size() +
                                   SelfOnInterfaceType::shmem_size() +
                                   CoupledOnInterfaceType::shmem_size())),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
          const int deg_start =
              team.league_rank() * num_edges; // edge0 is x^deg_start

          SelfDisplacementType self_disp(team.team_scratch(0));
          CoupledDisplacementType coupled_disp(team.team_scratch(0));
          SelfTransferType self_transfer(team);
          CoupledTransferType coupled_transfer(team);
          SelfOnInterfaceType self_on_interface(team.team_scratch(0));
          CoupledOnInterfaceType coupled_on_interface(team.team_scratch(0));

          // compute transfers. They will all be the same, so this test does not
          // check any transfer-function cross-contamination

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges), [&](const auto &iedge) {
                for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
                  for (int iintersection = 0;
                       iintersection < nquad_intersection; iintersection++) {
                    type_real eval = eval_lagrange(
                        edge_quadrature_points, ipoint,
                        intersection_quadrature_points[iintersection]);
                    self_transfer(iedge, ipoint, iintersection) = eval;
                    coupled_transfer(iedge, ipoint, iintersection) = eval;
                  }
                }
              });

          team.team_barrier();

          // populate fields with polynomials

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges), [&](const auto &iedge) {
                for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
                  for (int icomp = 0; icomp < SelfDisplacementType::components;
                       icomp++)
                    self_disp(iedge, ipoint, icomp) =
                        std::pow(edge_quadrature_points[ipoint],
                                 (deg_start + iedge + icomp) % deg_cap);
                  for (int icomp = 0;
                       icomp < CoupledDisplacementType::components; icomp++)
                    coupled_disp(iedge, ipoint, icomp) =
                        std::pow(edge_quadrature_points[ipoint],
                                 (deg_start + iedge + icomp) % deg_cap);
                }
              });

          team.team_barrier();

          // specfem::datatype::VectorPointViewType<type_real,
          // SelfDisplacementType::components, SelfDisplacementType::using_simd>

          // validate (self and coupled are independent, so we shouldn't need a
          // barrier in between them)

          specfem::algorithms::transfer(
              ChunkEdgeIndexSimulator<dimension_tag>(num_edges, team),
              self_transfer, self_disp,
              [&](const auto &index, const PointSelfDisplacementType &point) {
                for (int icomp = 0; icomp < SelfDisplacementType::components;
                     icomp++)
                  self_on_interface(index(0), index(1), icomp) = point(icomp);
              });
          specfem::algorithms::transfer(
              ChunkEdgeIndexSimulator<dimension_tag>(num_edges, team),
              coupled_transfer, coupled_disp,
              [&](const auto &index,
                  const PointCoupledDisplacementType &point) {
                for (int icomp = 0; icomp < CoupledDisplacementType::components;
                     icomp++)
                  coupled_on_interface(index(0), index(1), icomp) =
                      point(icomp);
              });

          team.team_barrier();

          // transfer should send polys to themselves, but in the new
          // basis. the expectation is just the intersection quadrature
          // point to the same power.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges), [&](const auto &iedge) {
                for (int iintersection = 0; iintersection < nquad_intersection;
                     iintersection++) {
                  const type_real x =
                      intersection_quadrature_points[iintersection];
                  for (int icomp = 0; icomp < SelfDisplacementType::components;
                       icomp++) {
                    const int deg = (deg_start + iedge + icomp) % deg_cap;
                    const type_real expect = std::pow(x, deg);
                    const type_real got =
                        self_on_interface(iedge, iintersection, icomp);
                    if (std::abs(expect - got) > tol) {
                      self_check_container(team.league_rank(), iedge) =
                          std::make_tuple(true, x, expect, got, deg, icomp,
                                          iintersection);
                    }
                  }
                  for (int icomp = 0;
                       icomp < CoupledDisplacementType::components; icomp++) {
                    const int deg = (deg_start + iedge + icomp) % deg_cap;
                    const type_real expect = std::pow(x, deg);
                    const type_real got =
                        coupled_on_interface(iedge, iintersection, icomp);
                    if (std::abs(expect - got) > tol) {
                      coupled_check_container(team.league_rank(), iedge) =
                          std::make_tuple(true, x, expect, got, deg, icomp,
                                          iintersection);
                    }
                  }
                }
              });
        });

    // host mirror to read failures

    CheckContainer::HostMirror h_self_check_container =
        Kokkos::create_mirror_view(self_check_container);
    CheckContainer::HostMirror h_coupled_check_container =
        Kokkos::create_mirror_view(coupled_check_container);
    Kokkos::deep_copy(h_self_check_container, self_check_container);
    Kokkos::deep_copy(h_coupled_check_container, coupled_check_container);

    for (int ileague = 0; ileague < num_iters; ileague++) {
      for (int iedge = 0; iedge < num_edges; iedge++) {
        {
          const auto [is_fail, x, expect, got, deg, icomp, iintersection] =
              h_self_check_container(ileague, iedge);
          if (is_fail) {
            // even though we already know its a fail, run EXPECT so we don't
            // need that boilerplate
            EXPECT_NEAR(got, expect, tol)
                << "Self side: Computing " << x << " ^ " << deg
                << " for iedge = " << iedge << ", icomp = " << icomp
                << " for intersection quadrature point " << iintersection;
          }
        }
        {
          const auto [is_fail, x, expect, got, deg, icomp, iintersection] =
              h_coupled_check_container(ileague, iedge);
          if (is_fail) {
            EXPECT_NEAR(got, expect, tol)
                << "Coupled side: Computing " << x << " ^ " << deg
                << " for iedge = " << iedge << ", icomp = " << icomp
                << " for intersection quadrature point " << iintersection;
          }
        }
      }
    }
  }
};

struct EdgeToInterfaceCouplingTestParams {
  std::string name;
  std::shared_ptr<EdgeToInterfaceParamsBase> params;

  void run_test() const { params->run_test(name); }

  template <specfem::dimension::type DimensionTag,
            specfem::interface::interface_tag InterfaceTag,
            std::size_t nquad_edge, std::size_t nquad_intersection,
            int num_edges =
                specfem::parallel_configuration::default_chunk_edge_config<
                    DimensionTag, Kokkos::DefaultExecutionSpace>::chunk_size>
  static EdgeToInterfaceCouplingTestParams
  from(const std::string &name, const type_real (&edge)[nquad_edge],
       const type_real (&intersection)[nquad_intersection]) {
    return { name,
             std::make_shared<
                 EdgeToInterfaceParams<num_edges, DimensionTag, InterfaceTag,
                                       nquad_edge, nquad_intersection> >(
                 edge, intersection) };
  }
};

class EdgeToInterfaceCouplingTest
    : public ::testing::TestWithParam<EdgeToInterfaceCouplingTestParams> {};

std::ostream &operator<<(std::ostream &os,
                         const EdgeToInterfaceCouplingTestParams &params) {
  os << params.name;
  return os;
}

TEST_P(EdgeToInterfaceCouplingTest, CouplingCalculation) {
  GetParam().run_test();
}

using init_type = std::tuple<std::string, EdgeToInterfaceParamsBase>;

INSTANTIATE_TEST_SUITE_P(
    NonconformingVariations, EdgeToInterfaceCouplingTest,
    ::testing::Values(EdgeToInterfaceCouplingTestParams::from<
                          specfem::dimension::type::dim2,
                          specfem::interface::interface_tag::elastic_acoustic>(
                          "dim2 elastic-acoustic", { -1, 1 }, { -0.5, 0, 0.5 }),
                      EdgeToInterfaceCouplingTestParams::from<
                          specfem::dimension::type::dim2,
                          specfem::interface::interface_tag::acoustic_elastic>(
                          "dim2 acoustic-elastic", { -1, -0.5, 0, 0.7, 1.2 },
                          { -0.3, 0, 0.4, 0.6 })));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
