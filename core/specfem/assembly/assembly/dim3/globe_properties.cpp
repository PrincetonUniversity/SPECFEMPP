#include "globe_properties.hpp"

#include "specfem/globe/elasticity.hpp"
#include "specfem/globe/model_evaluator.hpp"
#include "specfem/globe/region_codes.hpp"
#include "specfem/logger.hpp"
#include "specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp"
#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include "specfem/units.hpp"

#include "specfem/utilities/logarithmic_center.hpp"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void specfem::assembly::dim3_impl::read_globe_properties(
    const specfem::mesh::globe3d_mesh &input_mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  const auto start = std::chrono::steady_clock::now();
  std::size_t oracle_calls = 0;

  using Dimension = specfem::element::dimension_tag;
  using Medium = specfem::element::medium_tag;
  using Property = specfem::element::property_tag;

  const auto &element_types = assembly.element_types;
  if (!element_types.has_element_context()) {
    throw std::runtime_error("read_globe_properties: element_types carries no "
                             "globe element context");
  }

  // Only elastic isotropic elements have an attenuation container. Fail rather
  // than silently run anisotropic elements elastically (see issue #2059).
  if (input_mesh.attenuation.enabled &&
      element_types.get_elements_on_host(Medium::elastic, Property::anisotropic)
              .extent(0) > 0) {
    throw std::runtime_error(
        "read_globe_properties: attenuation is not supported for anisotropic "
        "elements");
  }

  const auto &globe = input_mesh.globe;
  if (!globe.planet_constants.has_value()) {
    throw std::runtime_error(
        "read_globe_properties: mesh carries no planet constants");
  }

  specfem::globe::ModelEvaluator evaluator(globe.model_config);
  evaluator.validate_database_constants(*globe.planet_constants,
                                        globe.model_verification.codes,
                                        globe.model_verification.flags);
  const auto evaluator_dims = evaluator.dimensions();
  if (evaluator_dims.ngllx != assembly.mesh.element_grid.ngllx ||
      evaluator_dims.nglly != assembly.mesh.element_grid.nglly ||
      evaluator_dims.ngllz != assembly.mesh.element_grid.ngllz) {
    throw std::runtime_error(
        "Globe model evaluator and mesh use different GLL dimensions");
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

  // One serial catalog call per element: the catalog retains element-scoped
  // Moho/sediment state, so batches must stay element-sized.
  for (int compute_ispec = 0; compute_ispec < assembly.mesh.nspec;
       ++compute_ispec) {
    const auto medium = element_types.get_medium_tag(compute_ispec);
    const auto property = element_types.get_property_tag(compute_ispec);
    if (!((medium == Medium::elastic && (property == Property::isotropic ||
                                         property == Property::anisotropic)) ||
          (medium == Medium::acoustic && property == Property::isotropic))) {
      throw std::runtime_error(
          "Unsupported globe material tags for compute element " +
          std::to_string(compute_ispec));
    }
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
        specfem::globe::to_region_code(
            element_types.get_region_tag(compute_ispec)),
        element_types.idoubling(compute_ispec),
        element_types.rmin(compute_ispec), element_types.rmax(compute_ispec),
        element_types.elem_in_crust(compute_ispec),
        element_types.elem_in_mantle(compute_ispec), xyz);

    ++oracle_calls;

    const bool tagged_anisotropic = property == Property::anisotropic;
    if (values.is_anisotropic && !tagged_anisotropic) {
      throw std::runtime_error(
          "Globe evaluator returned full anisotropy for a database element "
          "without the anisotropic property tag: compute element " +
          std::to_string(compute_ispec));
    }

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
            specfem::point::properties<specfem::tags::Tags<
                Dimension::dim3, Medium::acoustic, Property::isotropic, false>>
                point_property(1.0 / rho, kappa);
            specfem::assembly::store_on_host(index, point_property,
                                             assembly.properties);
          } else if (property == Property::anisotropic) {
            if (vs == 0.0) {
              throw std::runtime_error(
                  "Globe evaluator returned zero Vs for an elastic element");
            }
            specfem::globe::ensure_supported_azimuthal_anisotropy(
                values.gc_prime[ipoint], values.gs_prime[ipoint]);
            const auto &spherical = assembly.mesh.spherical_coordinates.h_coord;
            if (!values.is_anisotropic && spherical.data() == nullptr) {
              throw std::runtime_error(
                  "Globe anisotropic property build requires spherical "
                  "coordinates");
            }

            specfem::medium_physics::elasticity_tensor<double> model_cij{};
            const std::size_t cij_offset = 21 * ipoint;
            for (int component = 0; component < 21; ++component) {
              model_cij[component] = values.cij[cij_offset + component];
            }
            // Love parameters describe radial transverse isotropy. Convert
            // before any future attenuation shift: that shift must rotate
            // global -> radial, alter A/C/N/L, then rotate radial -> global.
            // Full anisotropy is already Cartesian and passes through
            // unchanged.
            const auto stiffness = specfem::globe::elasticity_from_model(
                values.is_anisotropic, model_cij, values.rho[ipoint],
                values.vpv[ipoint], values.vph[ipoint], values.vsv[ipoint],
                values.vsh[ipoint], values.eta[ipoint],
                values.is_anisotropic ? 0.0
                                      : spherical(compute_ispec, iz, iy, ix, 1),
                values.is_anisotropic
                    ? 0.0
                    : spherical(compute_ispec, iz, iy, ix, 2));

            specfem::point::properties<specfem::tags::Tags<
                Dimension::dim3, Medium::elastic, Property::anisotropic, false>>
                point_property(
                    stiffness[0], stiffness[1], stiffness[2], stiffness[3],
                    stiffness[4], stiffness[5], stiffness[6], stiffness[7],
                    stiffness[8], stiffness[9], stiffness[10], stiffness[11],
                    stiffness[12], stiffness[13], stiffness[14], stiffness[15],
                    stiffness[16], stiffness[17], stiffness[18], stiffness[19],
                    stiffness[20], rho);
            specfem::assembly::store_on_host(index, point_property,
                                             assembly.properties);
          } else {
            if (vs == 0.0) {
              throw std::runtime_error(
                  "Globe evaluator returned zero Vs for an elastic element");
            }
            const type_real mu = rho * vs * vs;
            const type_real kappa = rho * vp * vp - (4.0 / 3.0) * mu;
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
  Kokkos::fence();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  specfem::Logger::debug(
      [&](std::ostringstream &message) {
        message << "Globe property build [rank " << specfem::MPI::get_rank()
                << "]: " << oracle_calls << " oracle element calls, "
                << oracle_calls * npoints << " GLL points, " << elapsed << " s";
      },
      false);
}
