#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include <cstdio>
#include <string>

namespace specfem::test_configuration {

/**
 * @brief Globe assembly built only as far as the MPI interfaces.
 *
 * Mirrors the staging of the full ``assembly<dim3>`` constructor for a globe
 * mesh -- assembly mesh with reference anchors, element types with the globe
 * context, forward fields, then ``mpi<dim3>`` from the adjacency graph and
 * ``h_mesh_to_compute`` -- without the model evaluator, sources or receivers,
 * which the MPI interfaces do not depend on.
 */
struct GlobeMPI3D {
  constexpr static auto dimension = specfem::element::dimension_tag::dim3;

  specfem::mesh::globe3d_mesh mesh; ///< Globe mesh read for this rank
  specfem::assembly::mesh<dimension> assembly_mesh; ///< GLL points, iglob
                                                    ///< numbering, reordering
  specfem::assembly::element_types<dimension> element_types; ///< Per-element
                                                             ///< tags
  specfem::assembly::fields<dimension> fields; ///< Per-medium iglob numbering
  specfem::assembly::mpi<dimension> mpi_interfaces; ///< Exchange patterns

  /**
   * @brief Read this rank's database and build the MPI interfaces.
   *
   * @param databases_mpi Directory holding the mesher's
   *        ``proc??????_specfempp_database.bin`` files
   */
  explicit GlobeMPI3D(const std::string &databases_mpi)
      : mesh(specfem::io::read_globe_mesh(database_path(databases_mpi),
                                          specfem::attenuation::Setup{})) {
    const specfem::quadrature::quadratures quadratures(
        specfem::quadrature::gll::gll{});
    assembly_mesh = { mesh.nspec,
                      mesh.control_nodes.ngnod,
                      mesh.element_grid.ngllz,
                      mesh.element_grid.nglly,
                      mesh.element_grid.ngllx,
                      mesh.tags,
                      mesh.adjacency_graph,
                      mesh.control_nodes,
                      quadratures,
                      mesh.globe.reference_coordinates };
    element_types = { mesh.nspec, assembly_mesh.element_grid, assembly_mesh,
                      mesh.tags, mesh.globe.element_context };
    fields = { assembly_mesh, element_types,
               specfem::simulation::type::forward };
    mpi_interfaces = { mesh.adjacency_graph,
                       element_types,
                       specfem::simulation::type::forward,
                       fields,
                       assembly_mesh.h_mesh_to_compute,
                       mesh.element_grid.ngllz,
                       mesh.element_grid.nglly,
                       mesh.element_grid.ngllx };
  }

  /**
   * @brief This rank's database inside a globe ``DATABASES_MPI`` directory.
   *
   * @param databases_mpi Directory holding the mesher output
   * @return ``<databases_mpi>/proc<rank:06d>_specfempp_database.bin``
   */
  static std::string database_path(const std::string &databases_mpi) {
    char name[64];
    std::snprintf(name, sizeof(name), "proc%06d_specfempp_database.bin",
                  specfem::MPI::get_rank());
    return databases_mpi + "/" + name;
  }
};

} // namespace specfem::test_configuration
