#include "globe_properties.hpp"

#ifdef SPECFEM_HAS_GLOBE_MODEL
#include "specfem/globe_model.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include "specfem/utilities/logarithmic_center.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::dim3_impl {

void build_globe_properties(
    const specfem::mesh::mesh<specfem::element::dimension_tag::dim3>
        &input_mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  using Dimension = specfem::element::dimension_tag;
  using Medium = specfem::element::medium_tag;
  using Property = specfem::element::property_tag;

  if (!input_mesh.globe.has_value()) {
    return;
  }
  const auto &globe = *input_mesh.globe;
  const auto &stored_config = globe.model_config;

  specfem::globe_model::ModelConfig config;
  config.model_name = stored_config.model_name;
  config.planet_type = stored_config.planet_type;
  config.nchunks = stored_config.nchunks;
  config.nex_xi = stored_config.nex_xi;
  config.nex_eta = stored_config.nex_eta;
  config.ellipticity = globe.ellipticity;
  config.topography = globe.topography;
  config.oceans = globe.oceans;
  config.attenuation = globe.attenuation;
  config.gravity = globe.gravity;
  config.rotation = globe.rotation;
  config.min_attenuation_period = stored_config.min_attenuation_period;
  config.max_attenuation_period = stored_config.max_attenuation_period;

  specfem::globe_model::Evaluator evaluator(config);
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
  const int ngnod = input_mesh.control_nodes.ngnod;
  const std::size_t npoints = static_cast<std::size_t>(ngllz) * nglly * ngllx;
  std::vector<double> xyz(3 * npoints);

  const bool has_attenuation = input_mesh.attenuation.enabled;
  auto *attenuation_container =
      has_attenuation
          ? &assembly.attenuation
                 .get_container<Medium::elastic, Property::isotropic>()
          : nullptr;

  for (int compute_ispec = 0; compute_ispec < assembly.mesh.nspec;
       ++compute_ispec) {
    const int mesh_ispec = assembly.mesh.h_compute_to_mesh(compute_ispec);
    const auto &context = globe.element_context[mesh_ispec];
    std::size_t ipoint = 0;
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix, ++ipoint) {
          for (int component = 0; component < 3; ++component) {
            double coordinate = 0.0;
            for (int inode = 0; inode < ngnod; ++inode) {
              const int global_node =
                  input_mesh.control_nodes.control_node_index(mesh_ispec,
                                                              inode);
              coordinate += assembly.mesh.h_shape3D(iz, iy, ix, inode) *
                            globe.reference_coordinates(global_node, component);
            }
            xyz[3 * ipoint + component] = coordinate;
          }
        }
      }
    }

    const auto values = evaluator.evaluate_element(
        context.region, context.idoubling, context.rmin, context.rmax,
        context.element_in_crust, context.element_in_mantle, xyz);
    if (values.is_anisotropic) {
      throw std::runtime_error(
          "The globe evaluator returned anisotropic cij, but SPECFEM++ has no "
          "3-D anisotropic property container or kernel yet");
    }

    const auto medium =
        input_mesh.materials.material_index_mapping[mesh_ispec].type;
    ipoint = 0;
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
#endif
