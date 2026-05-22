#include "specfem/mesh.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_adjacency_graph.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_boundaries.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_control_nodes.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_materials.hpp"
#include "specfem/io/mesh/impl/fortran/dim3/read_pml_boundaries.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

specfem::mesh::mesh<specfem::element::dimension_tag::dim3>
specfem::io::read_3d_mesh(
    const std::string &database_file,
    const specfem::attenuation::Setup &attenuation_setup) {

  const bool attenuation_enabled = attenuation_setup.enabled;
  const auto &f0 = attenuation_setup.f0;
  const auto &band = attenuation_setup.band;
  // Read mesh parameters
  std::ifstream param_stream(database_file, std::ios::in | std::ios::binary);
  if (!param_stream.is_open()) {
    throw std::runtime_error("Could not open mesh parameters file: " +
                             database_file);
  }

  bool mesh_of_earth_chunk;
  specfem::io::fortran_read_line(param_stream, &mesh_of_earth_chunk);

  // TODO (Rohit: EARTH_CHUNK_MESH): Add support for mesh of earth chunk
  if (mesh_of_earth_chunk) {
    throw std::runtime_error(
        "Mesh of earth chunk is not supported in SPECFEM++ yet.");
  }

  specfem::mesh::mesh<specfem::element::dimension_tag::dim3> mesh;

  mesh.control_nodes =
      specfem::io::mesh::impl::fortran::dim3::read_control_nodes(param_stream);
  const auto [nspec, ngllz, nglly, ngllx, control_node_index, materials] =
      specfem::io::mesh::impl::fortran::dim3::read_materials(
          param_stream, mesh.control_nodes.ngnod, attenuation_enabled);
  mesh.nspec = nspec;
  mesh.element_grid.ngllz = ngllz;
  mesh.element_grid.nglly = nglly;
  mesh.element_grid.ngllx = ngllx;
  mesh.control_nodes.nspec = nspec;
  mesh.control_nodes.control_node_index = control_node_index;
  mesh.materials = materials;
  mesh.boundaries = specfem::io::mesh::impl::fortran::dim3::read_boundaries(
      param_stream, nspec, mesh.control_nodes);

  mesh.tags = specfem::mesh::tags<specfem::element::dimension_tag::dim3>(
      nspec, mesh.materials);

  // CPML boundaries are not supported yet
  // TODO (Rohit: PML_BOUNDARIES): Add support for PML boundaries
  specfem::io::mesh::impl::fortran::dim3::read_pml_boundaries(param_stream);

  // Read adjacency information
  mesh.adjacency_graph =
      specfem::io::mesh::impl::fortran::dim3::read_adjacency_graph(param_stream,
                                                                   mesh.nspec);

  param_stream.close();

  mesh.setup_coupled_interfaces();

  if (attenuation_enabled) {
    if (!f0.has_value()) {
      throw std::runtime_error(
          "Attenuation is enabled but no reference-frequency was provided in "
          "specfem_config.yaml. For 3D simulations, reference-frequency must "
          "be set explicitly under the 'attenuation' section.");
    }
    mesh.attenuation = {
      true, f0.value(), band,
      specfem::attenuation::compute_tau_sigma<specfem::constants::N_SLS>(band)
    };
    mesh.materials.apply_attenuation(mesh.attenuation);
  }

  return mesh;
}
