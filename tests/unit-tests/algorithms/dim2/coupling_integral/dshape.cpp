#include "specfem/algorithms/coupling_integral1d_dnshape.hpp"
#include "specfem/algorithms/shape_function_normal_derivative.hpp"

#include "specfem/chunk_edge.hpp"
#include "specfem/enums.hpp"

#include "utilities/include/fixture/impl/accessors.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"

#include "SPECFEM_Environment.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"
#include <gtest/gtest.h>

template <typename EdgeTypesView>
class ChunkEdgeIterator : public specfem::execution::TeamThreadRangePolicy<
                              Kokkos::TeamPolicy<>::member_type, int> {
private:
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

public:
  using base_type =
      specfem::execution::TeamThreadRangePolicy<KokkosIndexType, int>;
  using execution_space = typename base_type::execution_space;
  using index_type =
      specfem::execution::EdgePointIndex<specfem::element::dimension_tag::dim2,
                                         typename base_type::policy_index_type,
                                         execution_space>;

  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIterator(const EdgeTypesView &edge_types, const int &nedges,
                    const int &ngllz, const int &ngllx,
                    const KokkosIndexType &team_member)
      : edge_types(edge_types), nedges(nedges), ngllz(ngllz), ngllx(ngllz),
        num_points(std::max(ngllx, ngllz)),
        base_type(team_member, nedges * std::max(ngllx, ngllz)),
        global_offset(0) {}

  KOKKOS_INLINE_FUNCTION
  const index_type
  operator()(const typename base_type::policy_index_type &i) const {
    const int iedge = i % nedges;
    const int ipoint = i / nedges;

    const specfem::mesh_entity::dim2::type edge_type = edge_types(iedge);
    const bool is_leftright =
        (edge_type == specfem::mesh_entity::dim2::type::left ||
         edge_type == specfem::mesh_entity::dim2::type::right);
    const int num_points_norm = is_leftright ? ngllx : ngllz;

    const int inorm = (edge_type == specfem::mesh_entity::dim2::type::bottom ||
                       edge_type == specfem::mesh_entity::dim2::type::left)
                          ? 0
                          : num_points_norm - 1;

    const int iz = is_leftright ? ipoint : inorm;
    const int ix = is_leftright ? inorm : ipoint;

    return index_type(
        specfem::point::edge_index<specfem::element::dimension_tag::dim2>(
            global_offset + iedge /*ispec*/, global_offset + iedge /*iedge*/,
            ipoint, iz, ix, edge_type),
        iedge, i);
  }

  const int nedges;

private:
  EdgeTypesView edge_types;
  const int global_offset;
  const int ngllz;
  const int ngllx;
  const int num_points;
};

template <typename EdgeTypesView> class ChunkEdgeIndex {
public:
  static constexpr auto accessor_type =
      specfem::datatype::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;
  using iterator_type = ChunkEdgeIterator<EdgeTypesView>;

  /**
   * @brief Get Kokkos team member index.
   * @return Reference to Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Construct chunk edge index.
   * @param nedges Number of edges in chunk
   * @param kokkos_index Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndex(const EdgeTypesView &edge_types, const int &nedges,
                 const int &ngllz, const int &ngllx,
                 const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges),
        iterator(edge_types, nedges, ngllz, ngllx, kokkos_index) {}

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;                  ///< Number of edges in the chunk
  KokkosIndexType kokkos_index; /**< Kokkos team member for this chunk */
  iterator_type iterator;

public:
  KOKKOS_INLINE_FUNCTION const iterator_type &get_iterator() const {
    return iterator;
  }
};

/**
 * @brief Patches assembly::nonconforming_interfaces to not require mesh and
 * edge types.
 *
 * nonconforming_interfaces only provides const access to the impl containers.
 * This class grants access to resizing these containers directly. Note that
 * const access still allows modification of values inside the views.
 */
class nonconforming_interfaces_patch
    : public specfem::assembly::nonconforming_interfaces<
          specfem::element::dimension_tag::dim2> {
  int ngllz;
  int ngllx;
  int nquad_intersection;

public:
  nonconforming_interfaces_patch(const int &ngllz, const int &ngllx,
                                 const int &nquad_intersection)
      : ngllz(ngllz), ngllx(ngllx), nquad_intersection(nquad_intersection) {};

  template <specfem::element_coupling::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag,
            specfem::element_connections::type ConnectionTag>
  void reinit_container(const int &num_edges) {

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), CONNECTION_TAG(NONCONFORMING),
         INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
         BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                      COMPOSITE_STACEY_DIRICHLET)),
        CAPTURE(interface_container) {
          if constexpr (_interface_tag_ == InterfaceTag &&
                        _boundary_tag_ == BoundaryTag &&
                        _connection_tag_ == ConnectionTag) {
            _interface_container_ =
                InterfaceContainerType<_interface_tag_, _boundary_tag_,
                                       _connection_tag_>(
                    ngllz, ngllx, nquad_intersection, num_edges);
          }
        })
  }
};

// temporary test for purposes of uncombined coupling_integral
void execute_simple_dshape_test() {
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr auto interface_tag =
      specfem::element_coupling::interface_tag::acoustic_elastic;
  constexpr auto boundary_tag = specfem::element::boundary_tag::none;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // ==========================================================
  // test case (this will be replaced by fixtures eventually)

  using QuadX = specfem::test_fixture::QuadraturePoints::GLL2;
  using QuadZ = specfem::test_fixture::QuadraturePoints::GLL2;
  using QuadIntersection = specfem::test_fixture::QuadraturePoints::GL6;
  constexpr int num_edges = 1;

  const type_real intersection_min = -1;
  const type_real intersection_max = 1;
  const auto side = specfem::mesh_entity::dim2::type::top;
  // we are integrating d(shape_function)/dn * F, where
  // :: F(x) = x
  // :: n = [1, x] in the contravariant local edge basis (see
  //      coupling_integral1d_dnshape.hpp), or [x, 1] in typical local
  //      coordinates
  // :: shape functions on GLL2 x GLL1 elements (shape_function = L_xi(x) *
  // L_gamma(z))
  // :: integral on [-1,1] (intersection_<min/max>)
  // :: edge has same coordinates, and goes between [-1,1] in xi
  // :: solution: int(x * (x * L_xi'(x) * L_gamma(z) + L_xi(x) * L_gamma'(z)))

  // structs to use

  using LagrangeDerivative2D =
      specfem::test_fixture::LagrangeDerivative2D<QuadX, QuadZ>;
  LagrangeDerivative2D lagrange_derivative("dshape::lagrange_derivative");

  using IntersectionFunctionInitializer = specfem::test_fixture::
      IntersectionFunctionInitializer2D::FromAnalyticalFunction<
          specfem::test_fixture::AnalyticalFunctionType::Power<1>,
          QuadIntersection>;
  const auto function_view = specfem::test_fixture::IntersectionFunction2D(
                                 IntersectionFunctionInitializer())
                                 .get_view();

  // we will probably fixturize this at some point to make it not so bloated
  using IntersectionFactor =
      specfem::test_fixture::impl::NonconformingIntersectionFactorPatch<
          interface_tag, boundary_tag, num_edges, QuadIntersection::nquad>;
  auto intersection_factor = IntersectionFactor("dshape::intersection_factor");

  {
    const auto h_intersection_factor = intersection_factor.create_host_mirror();

    for (int iedge = 0; iedge < num_edges; iedge++) {
      for (int iquad = 0; iquad < QuadIntersection::nquad; iquad++) {
        h_intersection_factor(iedge, iquad) =
            specfem::test_fixture::QuadratureRule<QuadIntersection>::
                compute_lagrange_quadrature_weight(iquad, intersection_min,
                                                   intersection_max);
      }
    }
    intersection_factor.sync_to_device(h_intersection_factor);
  }

  // Contravariant normal derivative is not a datatype yet. Keep this for now.
  using IntersectionContraNormalFunction =
      specfem::test_fixture::AnalyticalFunctionType::Chain<
          specfem::test_fixture::AnalyticalFunctionType::Power<0>,
          specfem::test_fixture::AnalyticalFunctionType::Power<1> >;
  using IntersectionContraNormalInitializer =
      specfem::test_fixture::IntersectionFunctionInitializer2D::
          FromAnalyticalFunction<IntersectionContraNormalFunction,
                                 QuadIntersection>;
  const auto intersection_contra_normal =
      specfem::test_fixture::IntersectionFunction2D(
          IntersectionContraNormalInitializer())
          .get_view();

  // no data access help for transfer_function_self derivative, but the type
  // only needs to support deriv(iedge, iedge, iquad).
  using TransferFunctionDerivativeType =
      specfem::datatype::VectorChunkEdgeViewType<
          type_real, dimension_tag, num_edges, QuadX::nquad,
          QuadIntersection::nquad, false, memory_space,
          Kokkos::MemoryTraits<> >;

  TransferFunctionDerivativeType transfer_function_self_derivative(
      "dshape::transfer_function_self_derivative"); // init later with
                                                    // interface_container

  // ==========================================================

  constexpr int nquad_intersection = QuadIntersection::nquad;
  constexpr int ngllx = QuadX::nquad;
  constexpr int ngllz = QuadZ::nquad;
  constexpr int nquad_element = std::max(ngllx, ngllz);
  constexpr int chunk_size = 1;

  constexpr auto medium_self =
      specfem::element_coupling::attributes<dimension_tag,
                                            interface_tag>::self_medium();
  constexpr auto ncomp_self =
      specfem::element::attributes<dimension_tag, medium_self>::components;

  nonconforming_interfaces_patch nonconforming_interfaces(ngllz, ngllx,
                                                          nquad_intersection);
  nonconforming_interfaces.template reinit_container<
      interface_tag, boundary_tag,
      specfem::element_connections::type::nonconforming>(num_edges);

  const auto &interface_container =
      nonconforming_interfaces.template get_interface_container<
          interface_tag, boundary_tag,
          specfem::element_connections::type::nonconforming>();

  // =================================================================
  // populate this nonconforming interface container and
  // transfer_function_self_derivative
  {
    TransferFunctionDerivativeType::HostMirror h_tfsd =
        Kokkos::create_mirror_view(transfer_function_self_derivative);
    for (int iedge = 0; iedge < num_edges; iedge++) {
      for (int iquad_edge = 0; iquad_edge < ngllx; iquad_edge++) {
        for (int iquad_intersection = 0;
             iquad_intersection < nquad_intersection; iquad_intersection++) {

          // no transformation (it would go here, otherwise)
          const double intersection_point_in_edge_coords =
              QuadIntersection::quadrature_points[iquad_intersection];
          interface_container.h_transfer_function(iedge, iquad_intersection,
                                                  iquad_edge) = specfem::
              test_fixture::QuadratureRule<QuadX>::evaluate_lagrange_polynomial(
                  iquad_edge, intersection_point_in_edge_coords);
          h_tfsd(iedge, iquad_edge, iquad_intersection) = specfem::
              test_fixture::QuadratureRule<QuadX>::evaluate_lagrange_derivative(
                  iquad_edge, intersection_point_in_edge_coords);
        }
      }
    }
    Kokkos::deep_copy(transfer_function_self_derivative, h_tfsd);
    Kokkos::deep_copy(interface_container.transfer_function,
                      interface_container.h_transfer_function);
  }
  // =================================================================

  using EdgeTypesView = Kokkos::View<specfem::mesh_entity::dim2::type *,
                                     memory_space, Kokkos::MemoryTraits<> >;
  EdgeTypesView edge_types("dshape::edge_types", num_edges);
  EdgeTypesView::HostMirror h_edge_types =
      Kokkos::create_mirror_view(edge_types);
  for (int iedge = 0; iedge < num_edges; iedge++) {
    h_edge_types(iedge) = side;
  }
  Kokkos::deep_copy(edge_types, h_edge_types);

  // chunk subviews (FunctionType comes from copied test template -- skip the
  // rest and assume num_edges == 1)
  using FunctionType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, chunk_size, nquad_intersection, ncomp_self,
      false, memory_space, Kokkos::MemoryTraits<> >;

  const int num_chunks =
      num_edges / chunk_size + ((num_edges % chunk_size != 0) ? 1 : 0);

  Kokkos::View<type_real *[ngllz][ngllx], memory_space, Kokkos::MemoryTraits<> >
      computed_integrals("dshape::computed_integrals", num_edges);

  // Kokkos::View<type_real *[ngllz][ngllx], memory_space,
  // Kokkos::MemoryTraits<> >
  //     expected_solutions("dshape::expected_solutions", num_edges);
  // const auto h_expected_solutions =
  //     Kokkos::create_mirror_view(expected_solutions);
  Kokkos::View<type_real *[ngllz][ngllx], Kokkos::HostSpace>
      h_expected_solutions("dshape::h_expected_solutions", num_edges);

  // =================================================================
  // compute expected solutions

  using IntersectionQuadrature =
      specfem::test_fixture::QuadratureRule<QuadIntersection>;
  std::array<double, nquad_intersection> quadrature_weights;
  for (int iquad = 0; iquad < nquad_intersection; iquad++) {
    quadrature_weights[iquad] =
        IntersectionQuadrature::compute_lagrange_quadrature_weight(
            iquad, intersection_min, intersection_max);
  }

  // [!] modify for when num_edges > 1 in later test
  for (int iz = 0; iz < ngllz; iz++) {
    for (int ix = 0; ix < ngllx; ix++) {
      // use the intersection quadrature rule
      // solution: int(x * (x * L_xi'(x) * L_gamma(z) + L_xi(x) * L_gamma'(z)))
      double integral = 0;
      for (int iquad = 0; iquad < nquad_intersection; iquad++) {
        const double x = QuadIntersection::quadrature_points[iquad];
        const double z = 1; // since we are at the top
        const double dshapedxi =
            specfem::test_fixture::QuadratureRule<
                QuadX>::evaluate_lagrange_derivative(ix, x) *
            specfem::test_fixture::QuadratureRule<
                QuadZ>::evaluate_lagrange_polynomial(iz, z);
        const double dshapedga =
            specfem::test_fixture::QuadratureRule<
                QuadX>::evaluate_lagrange_polynomial(ix, x) *
            specfem::test_fixture::QuadratureRule<
                QuadZ>::evaluate_lagrange_derivative(iz, z);
        const double intersection_function =
            IntersectionFunctionInitializer::AnalyticalFunctionType::evaluate(
                x)[0];
        const auto n_contraedge = IntersectionContraNormalFunction::evaluate(x);
        double nxi;
        double nga;
        switch (side) {
        case specfem::mesh_entity::dim2::type::bottom:
          nga = -n_contraedge[0];
          nxi = n_contraedge[1];
          break;
        case specfem::mesh_entity::dim2::type::top:
          nga = n_contraedge[0];
          nxi = n_contraedge[1];
          break;
        case specfem::mesh_entity::dim2::type::left:
          nxi = -n_contraedge[0];
          nga = n_contraedge[1];
          break;
        case specfem::mesh_entity::dim2::type::right:
          nxi = n_contraedge[0];
          nga = n_contraedge[1];
          break;
        default:
          FAIL() << "Poorly posed test. \"side\" is not an edge!.";
        }
        integral +=
            quadrature_weights[iquad] *
            (intersection_function * (nxi * dshapedxi + nga * dshapedga));
      }
      h_expected_solutions(0, iz, ix) = (type_real)integral;
    }
  }

  // Kokkos::deep_copy(expected_solutions, h_expected_solutions);
  // =================================================================

  specfem::assembly::edge_types<dimension_tag> assembly_edge_types;
  specfem::assembly::mesh<dimension_tag> mesh;
  specfem::algorithms::shape_function_self_normal_derivatives<interface_tag,
                                                              boundary_tag>(
      assembly_edge_types, mesh, nonconforming_interfaces);

  Kokkos::parallel_for(
      "SimpleDShapeTest", Kokkos::TeamPolicy<>(num_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int chunk_index = team_member.league_rank();
        const FunctionType F(
            Kokkos::subview(function_view,
                            Kokkos::make_pair(chunk_index * chunk_size,
                                              (chunk_index + 1) * chunk_size),
                            Kokkos::ALL(), Kokkos::ALL()));
        specfem::algorithms::coupling_integral_dnshape(
            nonconforming_interfaces, ngllz, ngllx, lagrange_derivative,
            ChunkEdgeIndex(Kokkos::subview(edge_types,
                                           Kokkos::make_pair(
                                               chunk_index * chunk_size,
                                               (chunk_index + 1) * chunk_size)),
                           num_edges, ngllz, ngllx, team_member),
            F, intersection_factor, intersection_contra_normal,
            transfer_function_self_derivative,
            [&](const auto &index, const auto &point) {
              computed_integrals(index.ispec, index.iz, index.ix) = point(0);
            });
      });

  const auto h_computed_integrals = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), computed_integrals);

  for (int iedge = 0; iedge < num_edges; iedge++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {

        if (!specfem::utilities::is_close(
                h_computed_integrals(iedge, iz, ix),
                h_expected_solutions(iedge, iz, ix))) {
          ADD_FAILURE() << "Integral mismatch for edge " << iedge << "\n"
                        << "    at GLL point (iz = " << iz << ", ix = " << ix
                        << ")\n"
                        << "    expected: "
                        << h_expected_solutions(iedge, iz, ix) << "\n"
                        << "    computed: "
                        << h_computed_integrals(iedge, iz, ix);
        }
      }
    }
  }
}

TEST(CouplingIntegral, SimpleDShapeTest) { execute_simple_dshape_test(); }

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
