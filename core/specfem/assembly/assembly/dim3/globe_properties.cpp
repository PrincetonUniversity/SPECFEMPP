#include "globe_properties.hpp"

#include <stdexcept>

#include "specfem/globe/model_evaluator.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/logarithmic_center.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace specfem::assembly::dim3_impl {

void read_globe_properties(
    const specfem::mesh::globe3d_mesh &input_mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  using Dimension = specfem::element::dimension_tag;
  using Medium = specfem::element::medium_tag;
  using Property = specfem::element::property_tag;

  const auto iregion_code = [](const specfem::element::region_tag region) {
    switch (region) {
    case specfem::element::region_tag::crust_mantle:
      return 1;
    case specfem::element::region_tag::outer_core:
      return 2;
    case specfem::element::region_tag::inner_core:
      return 3;
    default:
      throw std::runtime_error("Unknown region tag for the globe evaluator");
    }
  };

  const auto &element_types = assembly.element_types;
  if (!element_types.has_element_context()) {
    throw std::runtime_error("read_globe_properties: element_types carries no "
                             "globe element context");
  }

  const auto &globe = input_mesh.globe;
  specfem::globe::ModelEvaluator evaluator(globe.model_config,
                                           globe.planet_constants);
  const auto evaluator_dims = evaluator.dimensions();
  if (evaluator_dims.ngllx != assembly.mesh.element_grid.ngllx ||
      evaluator_dims.nglly != assembly.mesh.element_grid.nglly ||
      evaluator_dims.ngllz != assembly.mesh.element_grid.ngllz) {
    throw std::runtime_error(
        "Globe model evaluator and mesh use different GLL dimensions");
  }
  globe.planet_constants.check_radii(evaluator.radii());

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int nglly = assembly.mesh.element_grid.nglly;
  const int ngllx = assembly.mesh.element_grid.ngllx;
  const int ngnod = input_mesh.control_nodes.ngnod;
  const std::size_t npoints = static_cast<std::size_t>(ngllz) * nglly * ngllx;
  const std::size_t ncoordinates = 3 * npoints;
  constexpr int max_batch_elements = 4096;
  std::vector<double> xyz(ncoordinates);
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> xyz_batch(
      "specfem::assembly::dim3::globe_reference_coordinates",
      std::min(max_batch_elements, assembly.mesh.nspec), ncoordinates);

  const auto h_compute_to_mesh = assembly.mesh.h_compute_to_mesh;
  const auto h_shape3D = assembly.mesh.h_shape3D;
  const auto control_node_index = input_mesh.control_nodes.control_node_index;
  const auto reference_coordinates = globe.reference_coordinates;

  const bool has_attenuation = input_mesh.attenuation.enabled;
  auto *attenuation_container =
      has_attenuation
          ? &assembly.attenuation
                 .get_container<Medium::elastic, Property::isotropic>()
          : nullptr;

  for (int batch_begin = 0; batch_begin < assembly.mesh.nspec;
       batch_begin += max_batch_elements) {
    const int batch_size =
        std::min(max_batch_elements, assembly.mesh.nspec - batch_begin);

    // The Fortran-backed globe evaluator is explicitly single-threaded, so only
    // the reference-coordinate interpolation is parallelized here.
    Kokkos::parallel_for(
        "specfem::assembly::dim3::interpolate_globe_reference_coordinates",
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                              Kokkos::Rank<5>>(
            { 0, 0, 0, 0, 0 }, { batch_size, ngllz, nglly, ngllx, 3 }),
        [batch_begin, control_node_index, h_compute_to_mesh, h_shape3D, ngllx,
         nglly, ngnod, reference_coordinates,
         xyz_batch](const int batch_ispec, const int iz, const int iy,
                    const int ix, const int component) {
          const int compute_ispec = batch_begin + batch_ispec;
          const int mesh_ispec = h_compute_to_mesh(compute_ispec);
          const std::size_t ipoint =
              (static_cast<std::size_t>(iz) * nglly + iy) * ngllx + ix;
          double coordinate = 0.0;
          for (int inode = 0; inode < ngnod; ++inode) {
            const int global_node = control_node_index(mesh_ispec, inode);
            coordinate += h_shape3D(iz, iy, ix, inode) *
                          reference_coordinates(global_node, component);
          }
          xyz_batch(batch_ispec, 3 * ipoint + component) = coordinate;
        });
    Kokkos::fence();

    for (int batch_ispec = 0; batch_ispec < batch_size; ++batch_ispec) {
      const int compute_ispec = batch_begin + batch_ispec;
      for (std::size_t icoordinate = 0; icoordinate < ncoordinates;
           ++icoordinate) {
        xyz[icoordinate] = xyz_batch(batch_ispec, icoordinate);
      }

      const auto property = element_types.get_property_tag(compute_ispec);
      const auto values = evaluator.evaluate_element(
          iregion_code(element_types.get_region_tag(compute_ispec)),
          element_types.idoubling(compute_ispec),
          element_types.rmin(compute_ispec), element_types.rmax(compute_ispec),
          element_types.elem_in_crust(compute_ispec),
          element_types.elem_in_mantle(compute_ispec), xyz);

      const bool tagged_anisotropic = property == Property::anisotropic;
      if (values.is_anisotropic != tagged_anisotropic) {
        throw std::runtime_error(
            "Globe evaluator anisotropy disagrees with the database property "
            "tag for compute element " +
            std::to_string(compute_ispec));
      }

      const auto medium = element_types.get_medium_tag(compute_ispec);
      std::size_t ipoint = 0;
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix, ++ipoint) {
            const type_real rho = values.rho[ipoint];
            const type_real vp = values.vp_iso[ipoint];
            const type_real vs = values.vs_iso[ipoint];
            const specfem::point::index<Dimension::dim3, false> index(
                compute_ispec, iz, iy, ix);
            if (medium == Medium::acoustic) {
              if (vs != 0.0) {
                throw std::runtime_error("Globe evaluator returned nonzero Vs "
                                         "for an acoustic element");
              }
              const type_real kappa = rho * vp * vp;
              specfem::point::properties<
                  specfem::tags::Tags<Dimension::dim3, Medium::acoustic,
                                      Property::isotropic, false>>
                  point_property(1.0 / rho, kappa);
              specfem::assembly::store_on_host(index, point_property,
                                               assembly.properties);
            } else if (property == Property::anisotropic) {
              if (vs == 0.0) {
                throw std::runtime_error(
                    "Globe evaluator returned zero Vs for an elastic element");
              }
              const std::size_t cij_offset = 21 * ipoint;
              specfem::point::properties<
                  specfem::tags::Tags<Dimension::dim3, Medium::elastic,
                                      Property::anisotropic, false>>
                  point_property(
                      values.cij[cij_offset + 0], values.cij[cij_offset + 1],
                      values.cij[cij_offset + 2], values.cij[cij_offset + 3],
                      values.cij[cij_offset + 4], values.cij[cij_offset + 5],
                      values.cij[cij_offset + 6], values.cij[cij_offset + 7],
                      values.cij[cij_offset + 8], values.cij[cij_offset + 9],
                      values.cij[cij_offset + 10], values.cij[cij_offset + 11],
                      values.cij[cij_offset + 12], values.cij[cij_offset + 13],
                      values.cij[cij_offset + 14], values.cij[cij_offset + 15],
                      values.cij[cij_offset + 16], values.cij[cij_offset + 17],
                      values.cij[cij_offset + 18], values.cij[cij_offset + 19],
                      values.cij[cij_offset + 20], rho);
              specfem::assembly::store_on_host(index, point_property,
                                               assembly.properties);
            } else {
              if (vs == 0.0) {
                throw std::runtime_error(
                    "Globe evaluator returned zero Vs for an elastic element");
              }
              const type_real mu = rho * vs * vs;
              const type_real kappa = rho * (vp * vp - (4.0 / 3.0) * vs * vs);
              specfem::point::properties<specfem::tags::Tags<
                  Dimension::dim3, Medium::elastic, Property::isotropic, false>>
                  point_property(kappa, mu, rho);
              specfem::assembly::store_on_host(index, point_property,
                                               assembly.properties);

              if (attenuation_container != nullptr) {
                const int attenuation_ispec =
                    compute_ispec -
                    attenuation_container->element_range.begin_index();
                attenuation_container->h_Qkappa(attenuation_ispec, iz, iy, ix) =
                    values.qkappa[ipoint];
                attenuation_container->h_Qmu(attenuation_ispec, iz, iy, ix) =
                    values.qmu[ipoint];
              }
            }
          }
        }
      }
    }
  }

  if (attenuation_container != nullptr) {
    using specfem::units::unit_symbols::Hz;
    const auto fc = specfem::utilities::logarithmic_center(
                        input_mesh.attenuation.band.min.raw(),
                        input_mesh.attenuation.band.max.raw()) *
                    Hz;
    const auto &elastic_properties =
        assembly.properties
            .get_container<Medium::elastic, Property::isotropic>();
    attenuation_container->recompute(
        elastic_properties, fc, input_mesh.attenuation.f0,
        input_mesh.attenuation.band, input_mesh.attenuation.tau_sigma);
  }
  assembly.properties.copy_to_device();
}

} // namespace specfem::assembly::dim3_impl
