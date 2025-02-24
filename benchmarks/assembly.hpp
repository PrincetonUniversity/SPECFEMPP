#pragma once

#include "IO/reader.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/fields/fields.hpp"
#include "compute/kernels/kernels.hpp"
#include "compute/properties/interface.hpp"
#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"

#include "impl_container.hpp"

namespace specfem {
/**
 * @brief Compute namespace defines data structures used to store data related
 * to finite element assembly.
 *
 * The data is organized in a manner that makes it effiecient to access when
 * computing finite element compute kernels.
 *
 */
namespace benchmarks {
/**
 * @brief Finite element assembly data
 *
 */
struct assembly {
  specfem::compute::mesh mesh; ///< Properties of the assembled mesh
  specfem::compute::element_types element_types; ///< Element tags for every
                                                 ///< spectral element
  specfem::compute::partial_derivatives partial_derivatives; ///< Partial
                                                             ///< derivatives of
                                                             ///< the basis
                                                             ///< functions
  specfem::benchmarks::assembly_properties properties; ///< Material properties
  specfem::compute::kernels kernels; ///< Frechet derivatives (Misfit kernels)
  specfem::compute::fields fields;

  assembly(const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
           const specfem::quadrature::quadratures &quadratures,
           const specfem::simulation::type simulation,
           const std::shared_ptr<specfem::IO::reader> &property_reader) {
    this->mesh = { mesh.tags, mesh.control_nodes, quadratures };
    this->element_types = { this->mesh.nspec, this->mesh.ngllz,
                            this->mesh.ngllx, this->mesh.mapping, mesh.tags };
    this->partial_derivatives = { this->mesh };
    this->properties = { this->mesh.nspec, this->mesh.ngllz,
                         this->mesh.ngllx, this->element_types,
                         mesh.materials,   property_reader != nullptr };
    this->kernels = { this->mesh.nspec, this->mesh.ngllz, this->mesh.ngllx,
                      this->element_types };
    this->fields = { this->mesh, this->element_types, simulation };
    return;
  }
};

} // namespace benchmarks
} // namespace specfem
