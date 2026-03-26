#include "specfem/algorithms/coupling_integral.hpp"

#include "specfem/chunk_edge.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/enums.hpp"

#include "utilities/include/builder/edgeview.hpp"
#include "utilities/include/builder/nonconforming_interfaces.hpp"
#include "utilities/include/fixture/impl/accessors.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"

#include "SPECFEM_Environment.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"
#include <gtest/gtest.h>

// temporary test for purposes of uncombined coupling_integral
void execute_simple_timesshape_test() {
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr auto interface_tag =
      specfem::element_coupling::interface_tag::acoustic_elastic;
  constexpr auto boundary_tag = specfem::element::boundary_tag::none;
  constexpr auto flux_scheme_tag =
      specfem::element_coupling::flux_scheme_tag::natural;
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
  // we are integrating shape_function * F, where
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
          interface_tag, boundary_tag, flux_scheme_tag, num_edges,
          QuadIntersection::nquad>;
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

  specfem::test_builder::NonconformingInterfacesPatch<dimension_tag>
      nonconforming_interfaces(ngllz, ngllx, nquad_intersection);
  nonconforming_interfaces.template reinit_container<
      interface_tag, boundary_tag,
      specfem::element_connections::type::nonconforming, flux_scheme_tag>(
      num_edges);

  const auto &interface_container =
      nonconforming_interfaces.template get_interface_container<
          interface_tag, boundary_tag,
          specfem::element_connections::type::nonconforming, flux_scheme_tag>();

  // =================================================================
  // populate this nonconforming interface container and
  // transfer_function_self_derivative
  {
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
        }
      }
    }
    Kokkos::deep_copy(interface_container.transfer_function,
                      interface_container.h_transfer_function);
  }
  // =================================================================

  auto edge_list_builder = specfem::test_builder::EdgeView(ngllz, ngllx)
                               .set_label("dshape::edgelist");
  for (int iedge = 0; iedge < num_edges; iedge++) {
    edge_list_builder.add_edge({ iedge, iedge, side, false });
  }
  const auto edgelist = edge_list_builder.build_on_device();

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
      // solution: int(x * L)
      double integral = 0;
      for (int iquad = 0; iquad < nquad_intersection; iquad++) {
        const double x = QuadIntersection::quadrature_points[iquad];
        const double z = 1; // since we are at the top
        const double shape = specfem::test_fixture::QuadratureRule<
                                 QuadX>::evaluate_lagrange_polynomial(ix, x) *
                             specfem::test_fixture::QuadratureRule<
                                 QuadZ>::evaluate_lagrange_polynomial(iz, z);
        const double intersection_function =
            IntersectionFunctionInitializer::AnalyticalFunctionType::evaluate(
                x)[0];
        integral += quadrature_weights[iquad] * (intersection_function * shape);
      }
      h_expected_solutions(0, iz, ix) = (type_real)integral;
    }
  }

  // Kokkos::deep_copy(expected_solutions, h_expected_solutions);
  // =================================================================

  constexpr int nquad_element_ = 3;
  constexpr int nquad_intersection_ = 7;
  if (nquad_element != nquad_element_) {
    throw std::runtime_error("dshape: Wrong kernel for nquad_element!");
  }
  if (nquad_intersection != nquad_intersection_) {
    throw std::runtime_error("dshape: Wrong kernel for nquad_intersection!");
  }

  using default_parallel_config =
      specfem::parallel_configuration::default_chunk_edge_config<
          dimension_tag, Kokkos::DefaultExecutionSpace>;
  // override parallel config to have test chunk size.
  using parallel_config = specfem::parallel_configuration::edge_chunk_config<
      dimension_tag, chunk_size, default_parallel_config::execution_space>;
  specfem::execution::ChunkedEdgeIterator chunk(parallel_config(), edgelist);

  specfem::execution::for_each_level(
      "specfem::compute::shape_function_normal_derivative", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &iter_chunk_index = chunk_iterator_index.get_index();
        const auto &team = iter_chunk_index.get_policy_index();
        const int &num_edges = iter_chunk_index.nedges();

        const int chunk_index = team.league_rank();
        const FunctionType F(
            Kokkos::subview(function_view,
                            Kokkos::make_pair(chunk_index * chunk_size,
                                              (chunk_index + 1) * chunk_size),
                            Kokkos::ALL(), Kokkos::ALL()));
        specfem::algorithms::coupling_integral(
            nonconforming_interfaces, iter_chunk_index, F, intersection_factor,
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

TEST(CouplingIntegral, SimpleTimesShapeTest) {
  execute_simple_timesshape_test();
}
