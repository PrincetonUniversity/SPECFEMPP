#include "specfem/assembly/boundaries/dim3/impl/injection/extract_injection_points.hpp"

#include "specfem/assembly/properties/load_on_host.hpp"
#include "specfem/point/index.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/tags.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::boundaries_impl::injection_impl {

// Read the local isotropic material (rho, vp, vs) at a compute GLL point. vs is
// 0 for acoustic. Throws on an unsupported medium/property combination.
void read_material(
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &properties,
    specfem::element::medium_tag medium,
    specfem::element::property_tag property, int ispec, int iz, int iy, int ix,
    type_real &rho, type_real &vp, type_real &vs) {
  const specfem::point::index<specfem::element::dimension_tag::dim3, false>
      index(ispec, iz, iy, ix);

  if (medium == specfem::element::medium_tag::elastic &&
      property == specfem::element::property_tag::isotropic) {
    specfem::point::properties<
        specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                            specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic,
                            specfem::element::attenuation_tag::none, false>>
        point;
    specfem::assembly::load_on_host(index, properties, point);
    rho = point.rho();
    vp = point.vp();
    vs = point.vs();
  } else if (medium == specfem::element::medium_tag::acoustic &&
             property == specfem::element::property_tag::isotropic) {
    specfem::point::properties<
        specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                            specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic,
                            specfem::element::attenuation_tag::none, false>>
        point;
    specfem::assembly::load_on_host(index, properties, point);
    rho = point.rho();
    vp = point.vp();
    vs = static_cast<type_real>(0);
  } else {
    std::ostringstream message;
    message << "extract_stacey_injection_points: unsupported medium/property "
               "combination at element "
            << ispec << "; only elastic/acoustic isotropic are supported";
    throw std::runtime_error(message.str());
  }
}

} // namespace specfem::assembly::boundaries_impl::injection_impl

specfem::injection::injection_points
specfem::assembly::boundaries_impl::extract_stacey_injection_points(
    const specfem::assembly::boundaries_impl::stacey<
        specfem::element::dimension_tag::dim3> &stacey_data,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::host_mirror_type
        &h_stacey_index_mapping,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &assembly_properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &assembly_element_types,
    const specfem::assembly::boundaries_impl::injection_boundary<
        specfem::element::dimension_tag::dim3>::PointIndexView::host_mirror_type
        &h_point_index) {
  const int nspec = static_cast<int>(h_stacey_index_mapping.extent(0));
  const int ngllz =
      static_cast<int>(stacey_data.h_quadrature_point_boundary_tag.extent(1));
  const int nglly =
      static_cast<int>(stacey_data.h_quadrature_point_boundary_tag.extent(2));
  const int ngllx =
      static_cast<int>(stacey_data.h_quadrature_point_boundary_tag.extent(3));
  const double zmax = static_cast<double>(assembly_mesh.zmax);

  Kokkos::deep_copy(h_point_index, -1);

  std::vector<type_real> xs, ys, zs, nxs, nys, nzs, rhos, vps, vss;
  std::vector<specfem::element::medium_tag> mediums;

  for (int ispec = 0; ispec < nspec; ++ispec) {
    const int local = h_stacey_index_mapping(ispec);
    if (local < 0) {
      continue;
    }
    const specfem::element::medium_tag medium =
        assembly_element_types.get_medium_tag(ispec);
    const specfem::element::property_tag property =
        assembly_element_types.get_property_tag(ispec);

    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix) {
          if (!(stacey_data.h_quadrature_point_boundary_tag(local, iz, iy,
                                                            ix) ==
                specfem::element::boundary_tag::stacey)) {
            continue;
          }

          // Coordinate view is indexed (ispec, iz, iy, ix, idim).
          const double x =
              static_cast<double>(assembly_mesh.h_coord(ispec, iz, iy, ix, 0));
          const double y =
              static_cast<double>(assembly_mesh.h_coord(ispec, iz, iy, ix, 1));
          const double z_mesh =
              static_cast<double>(assembly_mesh.h_coord(ispec, iz, iy, ix, 2));
          const double depth = zmax - z_mesh; // positive-down from the top

          // The stored face normal is J*grad(xi); normalize to a unit normal.
          double nx = static_cast<double>(
              stacey_data.h_face_normal(local, iz, iy, ix, 0));
          double ny = static_cast<double>(
              stacey_data.h_face_normal(local, iz, iy, ix, 1));
          double nz = static_cast<double>(
              stacey_data.h_face_normal(local, iz, iy, ix, 2));
          const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
          if (norm > 0.0) {
            nx /= norm;
            ny /= norm;
            nz /= norm;
          }

          type_real rho = 0;
          type_real vp = 0;
          type_real vs = 0;
          specfem::assembly::boundaries_impl::injection_impl::read_material(
              assembly_properties, medium, property, ispec, iz, iy, ix, rho, vp,
              vs);

          const int row = static_cast<int>(xs.size());
          xs.push_back(static_cast<type_real>(x));
          ys.push_back(static_cast<type_real>(y));
          zs.push_back(static_cast<type_real>(depth));
          nxs.push_back(static_cast<type_real>(nx));
          nys.push_back(static_cast<type_real>(ny));
          nzs.push_back(static_cast<type_real>(nz));
          mediums.push_back(medium);
          rhos.push_back(rho);
          vps.push_back(vp);
          vss.push_back(vs);
          h_point_index(local, iz, iy, ix) = row;
        }
      }
    }
  }

  const int n = static_cast<int>(xs.size());
  specfem::injection::injection_points points;
  points.x = Kokkos::View<type_real *, Kokkos::HostSpace>("injection_x", n);
  points.y = Kokkos::View<type_real *, Kokkos::HostSpace>("injection_y", n);
  points.z = Kokkos::View<type_real *, Kokkos::HostSpace>("injection_z", n);
  points.normal_x =
      Kokkos::View<type_real *, Kokkos::HostSpace>("injection_nx", n);
  points.normal_y =
      Kokkos::View<type_real *, Kokkos::HostSpace>("injection_ny", n);
  points.normal_z =
      Kokkos::View<type_real *, Kokkos::HostSpace>("injection_nz", n);
  points.medium =
      Kokkos::View<specfem::element::medium_tag *, Kokkos::HostSpace>(
          "injection_medium", n);
  points.rho = Kokkos::View<type_real *, Kokkos::HostSpace>("injection_rho", n);
  points.vp = Kokkos::View<type_real *, Kokkos::HostSpace>("injection_vp", n);
  points.vs = Kokkos::View<type_real *, Kokkos::HostSpace>("injection_vs", n);

  for (int i = 0; i < n; ++i) {
    points.x(i) = xs[i];
    points.y(i) = ys[i];
    points.z(i) = zs[i];
    points.normal_x(i) = nxs[i];
    points.normal_y(i) = nys[i];
    points.normal_z(i) = nzs[i];
    points.medium(i) = mediums[i];
    points.rho(i) = rhos[i];
    points.vp(i) = vps[i];
    points.vs(i) = vss[i];
  }

  return points;
}
