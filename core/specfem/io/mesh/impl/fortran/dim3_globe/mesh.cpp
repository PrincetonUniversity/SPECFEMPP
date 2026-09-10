#include "specfem/attenuation.hpp"
#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/common.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/read_adjacency_graph.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/read_boundaries.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/read_control_nodes.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/read_materials.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

specfem::mesh::globe3d_mesh specfem::io::read_globe_mesh(
    const std::string &database_file,
    const specfem::attenuation::Setup &attenuation_setup) {
  namespace reader = specfem::io::mesh::impl::fortran::dim3_globe;
  namespace reader_impl = specfem::io::mesh::impl::fortran::dim3_globe_impl;
  using Dimension = specfem::element::dimension_tag;

  std::ifstream stream(database_file, std::ios::in | std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("Could not open globe mesh database: " +
                             database_file);
  }

  const auto [magic, version] = reader_impl::read_magic(stream);
  if (magic != "SPECFEMPP_GLOBE_DB") {
    throw std::runtime_error("Not a SPECFEM++ globe mesh database: " +
                             database_file);
  }
  if (version < reader_impl::globe_database_version_min ||
      version > reader_impl::globe_database_version_max) {
    throw std::runtime_error("Unsupported globe mesh database version " +
                             std::to_string(version));
  }

  specfem::mesh::globe3d_mesh mesh;
  auto &globe = mesh.globe;
  globe.format_version = version;

  specfem::io::fortran_read_line(stream, &globe.model_config.planet_type,
                                 &globe.planet_radius, &globe.average_density);

  int ngnod = 0;
  specfem::io::fortran_read_line(stream, &ngnod, &mesh.element_grid.ngllx,
                                 &mesh.element_grid.nglly,
                                 &mesh.element_grid.ngllz, &globe.nregions);
  if (ngnod != 27) {
    throw std::runtime_error("Globe mesh database must contain hex27 anchors");
  }

  auto &model_config = globe.model_config;
  specfem::io::fortran_read_line(
      stream, &model_config.ellipticity, &model_config.topography,
      &model_config.gravity, &globe.full_gravity, &model_config.rotation,
      &model_config.attenuation, &model_config.oceans,
      &globe.has_reference_geometry);
  specfem::io::fortran_read_line(stream, &globe.material_mode);
  if (globe.material_mode != reader_impl::material_oracle) {
    throw std::runtime_error(
        "Only oracle-backed globe databases are supported");
  }

  model_config.model_name =
      reader_impl::read_fixed_string(stream, "model name");
  globe.model_verification.codes =
      reader_impl::read_counted_ints(stream, "model codes");
  globe.model_verification.flags =
      reader_impl::read_counted_logicals(stream, "model flags");
  specfem::io::fortran_read_line(stream, &model_config.nchunks,
                                 &model_config.nex_xi, &model_config.nex_eta);
  specfem::io::fortran_read_line(
      stream, &model_config.min_attenuation_period,
      &model_config.max_attenuation_period,
      &globe.model_verification.attenuation_source_frequency);
  model_config.validate();
  if (model_config.attenuation) {
    const double expected_source_frequency =
        1.0 / std::sqrt(model_config.min_attenuation_period *
                        model_config.max_attenuation_period);
    const double source_frequency_error =
        std::abs(globe.model_verification.attenuation_source_frequency -
                 expected_source_frequency);
    if (source_frequency_error >
        1.0e-12 * std::abs(expected_source_frequency)) {
      throw std::runtime_error(
          "Globe mesh database attenuation period band failed its central "
          "frequency check");
    }
  }

  const int nnode = reader::read_control_node_coordinates(stream, mesh, ngnod);
  const auto material_tags = reader::read_material_tags(stream, mesh);
  reader::read_control_node_indices(stream, mesh, ngnod, nnode);
  reader::read_boundaries(stream, mesh);
  reader::read_adjacency_graph(stream, mesh, nnode);
  reader_impl::check_stream(stream, "end of file");

  const bool attenuation_enabled =
      model_config.attenuation && attenuation_setup.enabled;
  mesh.materials =
      reader::make_materials(material_tags.medium_tags,
                             material_tags.property_tags, attenuation_enabled);
  mesh.tags = specfem::mesh::tags<Dimension::dim3>(
      mesh.nspec, mesh.materials, mesh.adjacency_graph, mesh.boundaries);

  mesh.setup_coupled_interfaces();
  if (attenuation_enabled) {
    if (!attenuation_setup.f0.has_value()) {
      throw std::runtime_error(
          "Globe attenuation requires an explicit reference frequency");
    }
    mesh.attenuation = {
      true, attenuation_setup.f0.value(), attenuation_setup.band,
      specfem::attenuation::compute_tau_sigma<specfem::constants::N_SLS>(
          attenuation_setup.band)
    };
    mesh.materials.apply_attenuation(mesh.attenuation);
  }

  return mesh;
}
