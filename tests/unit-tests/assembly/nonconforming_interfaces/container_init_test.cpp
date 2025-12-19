#include "../../SPECFEM_Environment.hpp"
#include "algorithms/locate_point.hpp"
#include "io/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/impl/compute_intersection.tpp"
#include "specfem/point.hpp"
#include "specfem/point/coordinates.hpp"
#include <gtest/gtest.h>
#include <memory>

template <typename TransferView, typename CoordView>
specfem::point::global_coordinates<specfem::dimension::type::dim2>
transfer_position_to_mortar(const TransferView &transfer_function,
                            const CoordView &coord_view, const int &ispec,
                            const specfem::mesh_entity::dim2::type &edge_type,
                            const int &iedge, const int &imortar) {
  specfem::point::global_coordinates<specfem::dimension::type::dim2> coords(0,
                                                                            0);
  const int ngllz = coord_view.extent(2);
  const int ngllx = coord_view.extent(3);
  const auto element = specfem::mesh_entity::element(ngllz, ngllx);

  for (int ipoint = 0; ipoint < transfer_function.extent(2); ipoint++) {
    const auto [iz, ix] = element.map_coordinates(edge_type, ipoint);
    const type_real transfer = transfer_function(iedge, imortar, ipoint);
    coords.x += transfer * coord_view(0, ispec, iz, ix);
    coords.z += transfer * coord_view(1, ispec, iz, ix);
  }
  return coords;
}

type_real estimate_intersection_length(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &icoorg,
    const int &ispec, const specfem::mesh_entity::dim2::type &iedge,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &jcoorg,
    const int &jspec, const specfem::mesh_entity::dim2::type &jedge) {
  const int num_segments = 20;
  const int ngnod = icoorg.extent(0);
  const Kokkos::View<type_real *, Kokkos::HostSpace> mortar_quad("mortar_quad",
                                                                 2);
  mortar_quad(0) = -1;
  mortar_quad(1) = 1;
  auto intersection_localcoords =
      specfem::assembly::nonconforming_interfaces_impl::compute_intersection(
          icoorg, jcoorg, iedge, jedge, mortar_quad);
  type_real intersect_lo = std::min(intersection_localcoords[0].first,
                                    intersection_localcoords[1].first);
  type_real intersect_hi = std::max(intersection_localcoords[0].first,
                                    intersection_localcoords[1].first);

  type_real xi, gamma;
  auto &edgecoord = [&xi, &gamma, &iedge]() -> type_real & {
    if (iedge == specfem::mesh_entity::dim2::type::bottom) {
      gamma = -1;
      return xi;
    } else if (iedge == specfem::mesh_entity::dim2::type::right) {
      xi = 1;
      return gamma;
    } else if (iedge == specfem::mesh_entity::dim2::type::top) {
      gamma = 1;
      return xi;
    } else {
      xi = -1;
      return gamma;
    }
  }();
  type_real total_length = 0;
  for (int iseg = 0; iseg < num_segments; iseg++) {
    edgecoord =
        intersect_lo + (intersect_hi - intersect_lo) * iseg / num_segments;
    auto glob_lo =
        specfem::jacobian::compute_locations(icoorg, ngnod, xi, gamma);
    edgecoord = intersect_lo +
                (intersect_hi - intersect_lo) * (iseg + 1) / num_segments;
    auto glob_hi =
        specfem::jacobian::compute_locations(icoorg, ngnod, xi, gamma);
    total_length +=
        std::sqrt((glob_hi.x - glob_lo.x) * (glob_hi.x - glob_lo.x) +
                  (glob_hi.z - glob_lo.z) * (glob_hi.z - glob_lo.z));
  }
  return total_length;
}

void estimate_verify_normal(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int &ispec, const specfem::mesh_entity::dim2::type &iedge,
    const type_real &edgecoord, const type_real &normal_x,
    const type_real &normal_z) {
  const type_real h = 1e-3;
  const type_real eps = 1e-2;
  const int ngnod = coorg.extent(0);

  // full local coords
  type_real xi, gamma;
  specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
      jacobian;

  /* coordinate of edge, references either xi or gamma. Other coord is
   * edge-constrained.
   *
   * Additionally, we can reference the coordinate of the jacobian matrix.
   *
   * by orth, here we mean the coordinate opposite the edge coordinate (orth in
   * local coords, not global)
   */
  auto [orthcoord, jacobian_edgecoordx, jacobian_edgecoordz,
        jacobian_orthcoordx, jacobian_orthcoordz] =
      [&xi, &gamma, &iedge,
       &jacobian]() -> std::tuple<type_real &, type_real &, type_real &,
                                  type_real &, type_real &> {
    if (iedge == specfem::mesh_entity::dim2::type::bottom) {
      gamma = -1;
      return { gamma, jacobian.xix, jacobian.xiz, jacobian.gammax,
               jacobian.gammaz };
    } else if (iedge == specfem::mesh_entity::dim2::type::right) {
      xi = 1;
      return { xi, jacobian.gammax, jacobian.gammaz, jacobian.xix,
               jacobian.xiz };
    } else if (iedge == specfem::mesh_entity::dim2::type::top) {
      gamma = 1;
      return { gamma, jacobian.xix, jacobian.xiz, jacobian.gammax,
               jacobian.gammaz };
    } else {
      xi = -1;
      return { xi, jacobian.gammax, jacobian.gammaz, jacobian.xix,
               jacobian.xiz };
    }
  }();
  jacobian = specfem::jacobian::compute_jacobian(coorg, ngnod, xi, gamma);
  EXPECT_TRUE(std::abs(jacobian_edgecoordx * normal_x +
                       jacobian_edgecoordz * normal_z) < eps)
      << "Given normal (" << normal_x << ", " << normal_z
      << ") is not perpendicular to edge tangent vector ("
      << jacobian_edgecoordx << ", " << jacobian_edgecoordz << ") on spec "
      << ispec << " edge " << specfem::mesh_entity::dim2::to_string(iedge)
      << " at local coordinate " << edgecoord << ".";

  // dot with dx/dorthcoord. at orthcoord (+/- 1), this would be outward facing
  EXPECT_TRUE(normal_x * (orthcoord * jacobian_orthcoordx) +
                  normal_z * (orthcoord * jacobian_orthcoordz) >
              0)
      << "Given normal (" << normal_x << ", " << normal_z
      << ") is not outward facing on spec " << ispec << " edge "
      << specfem::mesh_entity::dim2::to_string(iedge) << " at local coordinate "
      << edgecoord << ".";

  EXPECT_TRUE(std::abs(normal_x * normal_x + normal_z * normal_z - 1) < eps)
      << "Given normal (" << normal_x << ", " << normal_z
      << ") is not unit length on spec " << ispec << " edge "
      << specfem::mesh_entity::dim2::to_string(iedge) << " at local coordinate "
      << edgecoord << ".";
}

void test_nonconforming_container_transfers(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {
  const type_real length_verify_epsilon = 1e-2;

  const int ngllx = assembly.mesh.element_grid.ngllx;
  const int ngllz = assembly.mesh.element_grid.ngllz;

  const auto &nc_interface_acoustic_elastic =
      assembly.nonconforming_interfaces.get_interface_container<
          specfem::interface::interface_tag::acoustic_elastic,
          specfem::element::boundary_tag::none,
          specfem::connections::type::nonconforming>();

  const auto &nc_interface_elastic_acoustic =
      assembly.nonconforming_interfaces.get_interface_container<
          specfem::interface::interface_tag::elastic_acoustic,
          specfem::element::boundary_tag::none,
          specfem::connections::type::nonconforming>();

  const auto [acoustic_edges, elastic_edges] =
      assembly.edge_types.get_edges_on_host(
          specfem::connections::type::nonconforming,
          specfem::interface::interface_tag::acoustic_elastic,
          specfem::element::boundary_tag::none);

  const int nedges = acoustic_edges.n_edges;
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function.extent(0), nedges)
      << "acoustic side of the the acoustic-elastic interface does not have "
         "the correct number of intersections.";
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function_other.extent(0),
            nedges)
      << "elastic side of the the acoustic-elastic interface does not have the "
         "correct number of intersections.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function_other.extent(0),
            nedges)
      << "acoustic side of the the elastic-acoustic interface does not have "
         "the correct number of intersections.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function.extent(0), nedges)
      << "elastic side of the the elastic-acoustic interface does not have the "
         "correct number of intersections.";

  const int ngll = ngllz;
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function.extent(2), ngll)
      << "acoustic side of the the acoustic-elastic interface does not have "
         "the correct number of edge quadrature points.";
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function_other.extent(2),
            ngll)
      << "elastic side of the the acoustic-elastic interface does not have the "
         "correct number of edge quadrature points.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function_other.extent(2),
            ngll)
      << "acoustic side of the the elastic-acoustic interface does not have "
         "the correct number of edge quadrature points.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function.extent(2), ngll)
      << "elastic side of the the elastic-acoustic interface does not have the "
         "correct number of edge quadrature points.";

  // properly set this once we get mortar quad parameter up
  const int nquad_mortar =
      nc_interface_acoustic_elastic.h_transfer_function.extent(1);
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function.extent(1),
            nquad_mortar)
      << "acoustic side of the the acoustic-elastic interface does not have "
         "the correct interface quadrature size.";
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function_other.extent(1),
            nquad_mortar)
      << "elastic side of the the acoustic-elastic interface does not have the "
         "correct interface quadrature size.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function_other.extent(1),
            nquad_mortar)
      << "acoustic side of the the elastic-acoustic interface does not have "
         "the correct interface quadrature size.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function.extent(1),
            nquad_mortar)
      << "elastic side of the the elastic-acoustic interface does not have the "
         "correct interface quadrature size.";

  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      ac_coorg("ac_coorg", assembly.mesh.ngnod);
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      el_coorg("el_coorg", assembly.mesh.ngnod);

  for (int iedge = 0; iedge < nedges; iedge++) {
    const int ac_spec = acoustic_edges(iedge).element_index;
    const specfem::mesh_entity::dim2::type ac_edgetype =
        acoustic_edges(iedge).edge_type;
    const int el_spec = elastic_edges(iedge).element_index;
    const specfem::mesh_entity::dim2::type el_edgetype =
        elastic_edges(iedge).edge_type;
    for (int i = 0; i < assembly.mesh.ngnod; i++) {
      ac_coorg(i).x = assembly.mesh.h_control_node_coord(0, ac_spec, i);
      ac_coorg(i).z = assembly.mesh.h_control_node_coord(1, ac_spec, i);
      el_coorg(i).x = assembly.mesh.h_control_node_coord(0, el_spec, i);
      el_coorg(i).z = assembly.mesh.h_control_node_coord(1, el_spec, i);
    }

    // verify that the transfer function weights sum to the intersection
    // length (integrate 1 ds)
    type_real intersection_length = estimate_intersection_length(
        ac_coorg, ac_spec, ac_edgetype, el_coorg, el_spec, el_edgetype);
    type_real ac_weight_sum = 0;
    type_real el_weight_sum = 0;
    for (int imortar = 0; imortar < nquad_mortar; imortar++) {

      // interpolated positions on acoustic-elastic interface
      const specfem::point::global_coordinates<specfem::dimension::type::dim2>
          acel_self = transfer_position_to_mortar(
              nc_interface_acoustic_elastic.h_transfer_function,
              assembly.mesh.h_coord, ac_spec, ac_edgetype, iedge, imortar);
      const specfem::point::global_coordinates<specfem::dimension::type::dim2>
          acel_other = transfer_position_to_mortar(
              nc_interface_acoustic_elastic.h_transfer_function_other,
              assembly.mesh.h_coord, el_spec, el_edgetype, iedge, imortar);
      // interpolated positions on elastic-acoustic interface
      const specfem::point::global_coordinates<specfem::dimension::type::dim2>
          elac_self = transfer_position_to_mortar(
              nc_interface_elastic_acoustic.h_transfer_function,
              assembly.mesh.h_coord, el_spec, el_edgetype, iedge, imortar);
      const specfem::point::global_coordinates<specfem::dimension::type::dim2>
          elac_other = transfer_position_to_mortar(
              nc_interface_elastic_acoustic.h_transfer_function_other,
              assembly.mesh.h_coord, ac_spec, ac_edgetype, iedge, imortar);
      EXPECT_TRUE(specfem::point::distance(acel_self, acel_other) < 1e-3)
          << "acoustic-elastic interface has transfer functions from either "
             "side mapping to incompatible coordinates.";
      EXPECT_TRUE(specfem::point::distance(elac_self, elac_other) < 1e-3)
          << "elastic-acoustic interface has transfer functions from either "
             "side mapping to incompatible coordinates.";
      EXPECT_TRUE(specfem::point::distance(acel_self, elac_self) < 1e-3)
          << "elastic-acoustic and acoustic-elastic interface do not agree in "
             "coordinates.";

      ac_weight_sum +=
          nc_interface_acoustic_elastic.h_intersection_factor(iedge, imortar);
      el_weight_sum +=
          nc_interface_elastic_acoustic.h_intersection_factor(iedge, imortar);

      type_real ac_coord = 0;
      type_real el_coord = 0;
      // compute edge coordinates by interpolating x (transfer id function)
      for (int ipoint = 0; ipoint < ngll; ipoint++) {
        ac_coord += nc_interface_acoustic_elastic.h_transfer_function(
                        iedge, imortar, ipoint) *
                    assembly.mesh.h_xi(ipoint);
        ac_coord += nc_interface_elastic_acoustic.h_transfer_function(
                        iedge, imortar, ipoint) *
                    assembly.mesh.h_xi(ipoint);
      }

      // verify interface normals
      estimate_verify_normal(
          ac_coorg, ac_spec, ac_edgetype, ac_coord,
          nc_interface_acoustic_elastic.h_intersection_normal(iedge, imortar,
                                                              0),
          nc_interface_acoustic_elastic.h_intersection_normal(iedge, imortar,
                                                              1));
      estimate_verify_normal(
          el_coorg, el_spec, el_edgetype, el_coord,
          nc_interface_elastic_acoustic.h_intersection_normal(iedge, imortar,
                                                              0),
          nc_interface_elastic_acoustic.h_intersection_normal(iedge, imortar,
                                                              1));
    }
    EXPECT_TRUE(std::abs(ac_weight_sum - intersection_length) < 1e-3)
        << "acoustic-elastic interface has transfer functions with weights "
           "that do not sum to the intersection length. Weights sum to "
        << ac_weight_sum << " but intersection length is "
        << intersection_length << ".";
    EXPECT_TRUE(std::abs(el_weight_sum - intersection_length) < 1e-3)
        << "elastic-acoustic interface has transfer functions with weights "
           "that do not sum to the intersection length. Weights sum to "
        << el_weight_sum << " but intersection length is "
        << intersection_length << ".";

    // test if edge_normal is correct
  }
}

TEST(NonconformingInterfaces, ContainerInitialization) {
  std::string database_file("data/dim2/3_elem_nonconforming/database.bin");
  specfem::MPI::MPI *mpi = SPECFEMEnvironment::get_mpi();

  const auto mesh = specfem::io::read_2d_mesh(
      database_file, specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te, mpi);

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::dimension::type::dim2> > >
      sources;
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim2> > >
      receivers;
  specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
      mesh, quadrature, sources, receivers, {}, 1.0, 0.0, 1, 1, 1,
      specfem::simulation::type::forward, false, nullptr);

  // consider adding more assemblies to this test
  test_nonconforming_container_transfers(assembly);
}
