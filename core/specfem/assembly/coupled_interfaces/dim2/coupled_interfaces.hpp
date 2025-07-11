#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/coupled_interfaces/interface_container.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"

namespace specfem::assembly {
/**
 * @brief Information about coupled interfaces between 2 different media
 *
 */
template <> struct coupled_interfaces<specfem::dimension::type::dim2> {

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension of spectral
                                      ///< elements
  ///@}

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
   * @param mesh_assembly Mesh information for assembly
   * @param jacobian_matrix Jacobian matrix for every quadrature point
   * @param properties Material properties for every quadrature point
   */
  coupled_interfaces(
      const specfem::mesh::mesh<dimension_tag> &mesh,
      const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
      const specfem::assembly::jacobian_matrix &jacobian_matrix,
      const specfem::assembly::element_types &element_types);
  ///@}

  /**
   * @brief Get the interface container that contains information about the
   * interface between medium1 and medium2
   *
   * @tparam medium1 Self medium of the interface
   * @tparam medium2 Other medium of the interface
   * @return specfem::assembly::interface_container<medium1, medium2> Interface
   * container
   */
  template <specfem::element::medium_tag medium1,
            specfem::element::medium_tag medium2>
  specfem::assembly::interface_container<dimension_tag, medium1, medium2>
  get_interface_container() const;

  specfem::assembly::interface_container<
      dimension_tag, specfem::element::medium_tag::elastic_psv,
      specfem::element::medium_tag::acoustic>
      elastic_acoustic; ///< Elastic-acoustic interface

  specfem::assembly::interface_container<
      dimension_tag, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic; ///< Acoustic-poroelastic interface

  specfem::assembly::interface_container<
      dimension_tag, specfem::element::medium_tag::elastic_psv,
      specfem::element::medium_tag::poroelastic>
      elastic_poroelastic; ///< Elastic-poroelastic interface
};
} // namespace specfem::assembly
