#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/coupled_interfaces/interface_container.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/interface.hpp"
#include "interface_container.hpp"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"

namespace specfem {
namespace compute {
/**
 * @brief Information about coupled interfaces between 2 different media
 *
 */
struct coupled_interfaces {

  /**
   * @brief Default constructor
   *
   */
  coupled_interfaces() = default;

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Compute coupled interfaces for a given mesh
   *
   * @param mesh Finite element mesh information
   * @param points Information on finite element assembly
   * @param quadrature Quadrature information
   * @param partial_derivatives Partial derivatives for every quadrature point
   * @param properties Material properties for every quadrature point
   * @param mapping Mapping between mesh and compute spectral element indexing
   */
  coupled_interfaces(
      const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
      const specfem::compute::points &points,
      const specfem::compute::quadrature &quadrature,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types,
      const specfem::compute::mesh_to_compute_mapping &mapping);
  ///@}

  /**
   * @brief Get the interface container that contains information about the
   * interface between medium1 and medium2
   *
   * @tparam medium1 Self medium of the interface
   * @tparam medium2 Other medium of the interface
   * @return specfem::compute::interface_container<medium1, medium2> Interface
   * container
   */
  template <specfem::element::medium_tag medium1,
            specfem::element::medium_tag medium2>
  specfem::compute::interface_container<medium1, medium2>
  get_interface_container() const;

  specfem::compute::interface_container<
      specfem::element::medium_tag::elastic_sv,
      specfem::element::medium_tag::acoustic>
      elastic_acoustic; ///< Elastic-acoustic interface

  specfem::compute::interface_container<
      specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic; ///< Acoustic-poroelastic interface

  specfem::compute::interface_container<
      specfem::element::medium_tag::elastic_sv,
      specfem::element::medium_tag::poroelastic>
      elastic_poroelastic; ///< Elastic-poroelastic interface
};
} // namespace compute
} // namespace specfem
