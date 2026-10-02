#include "globe_properties.hpp"

#include <stdexcept>

#include "specfem/globe_model.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include "specfem/utilities/logarithmic_center.hpp"
#include <cmath>
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

  const auto &element_types = assembly.element_types;
  if (!element_types.has_element_context()) {
    throw std::runtime_error("read_globe_properties: element_types carries no "
                             "globe element context");
  }

  const auto &globe = input_mesh.globe;
  specfem::globe_model::Evaluator evaluator(globe.model_config);
  const auto evaluator_dims = evaluator.dims();
  if (evaluator_dims.ngllx != assembly.mesh.element_grid.ngllx ||
      evaluator_dims.nglly != assembly.mesh.element_grid.nglly ||
      evaluator_dims.ngllz != assembly.mesh.element_grid.ngllz) {
    throw std::runtime_error(
        "Globe model evaluator and mesh use different GLL dimensions");
  }
  const auto scales = evaluator.scales();
  if (std::abs(scales.length - globe.planet_radius) >
      1.0e-10 * globe.planet_radius) {
    throw std::runtime_error(
        "Globe model evaluator and mesh database use different planet radii");
  }

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int nglly = assembly.mesh.element_grid.nglly;
  const int ngllx = assembly.mesh.element_grid.ngllx;
  const std::size_t npoints = static_cast<std::size_t>(ngllz) * nglly * ngllx;
  std::vector<double> xyz(3 * npoints);

  // Reference (undeformed) coordinates when the database provides them;
  // aliases the final coordinates otherwise.
  const auto &h_sampling_coord = assembly.mesh.h_reference_coord;

  const bool has_attenuation = input_mesh.attenuation.enabled;
  auto *attenuation_container =
      has_attenuation
          ? &assembly.attenuation
                 .get_container<Medium::elastic, Property::isotropic>()
          : nullptr;

  // The Fortran-backed globe evaluator is explicitly single-threaded.
  for (int compute_ispec = 0; compute_ispec < assembly.mesh.nspec;
       ++compute_ispec) {
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix) {
          const std::size_t ipoint =
              (static_cast<std::size_t>(iz) * nglly + iy) * ngllx + ix;
          xyz[3 * ipoint + 0] = h_sampling_coord(compute_ispec, iz, iy, ix, 0);
          xyz[3 * ipoint + 1] = h_sampling_coord(compute_ispec, iz, iy, ix, 1);
          xyz[3 * ipoint + 2] = h_sampling_coord(compute_ispec, iz, iy, ix, 2);
        }
      }
    }

    const auto values = evaluator.evaluate_element(
        element_types.get_region_tag(compute_ispec),
        element_types.idoubling(compute_ispec),
        element_types.rmin(compute_ispec), element_types.rmax(compute_ispec),
        element_types.elem_in_crust(compute_ispec),
        element_types.elem_in_mantle(compute_ispec), xyz);

    const bool tagged_anisotropic =
        element_types.get_property_tag(compute_ispec) == Property::anisotropic;
    if (values.is_anisotropic != tagged_anisotropic) {
      throw std::runtime_error(
          "Globe evaluator anisotropy disagrees with the database property "
          "tag for compute element " +
          std::to_string(compute_ispec));
    }
    if (tagged_anisotropic) {
      // Storing the oracle's 21 cij in the anisotropic container is issue
      // #2043; the evaluator's Voigt order matches the container's.
      throw std::runtime_error(
          "Storing anisotropic globe properties from the evaluator is not "
          "implemented yet");
    }

    const auto medium = element_types.get_medium_tag(compute_ispec);
    std::size_t ipoint = 0;
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix, ++ipoint) {
          const type_real rho = values.rho[ipoint] * scales.density;
          const type_real vp = values.vp_iso[ipoint] * scales.velocity;
          const type_real vs = values.vs_iso[ipoint] * scales.velocity;
          const specfem::point::index<Dimension::dim3, false> index(
              compute_ispec, iz, iy, ix);
          if (medium == Medium::acoustic) {
            if (vs != 0.0) {
              throw std::runtime_error("Globe evaluator returned nonzero Vs "
                                       "for an acoustic element");
            }
            const type_real kappa = rho * vp * vp;
            specfem::point::properties<specfem::tags::Tags<
                Dimension::dim3, Medium::acoustic, Property::isotropic, false>>
                point_property(1.0 / rho, kappa);
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
