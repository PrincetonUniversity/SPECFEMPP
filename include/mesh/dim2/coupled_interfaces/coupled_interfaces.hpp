#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "interface_container.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_mpi/specfem_mpi.hpp"
#include <fstream>
#include <variant>

namespace specfem {
namespace mesh {

// TODO(Rohit: ADJ_GRAPH_DEFAULT)
// Remove coupled_interfaces functionality when adjacency graph is the default
// Coupled interfaces should be saved as weakly conforming adjacency graph edges
/**
 * @brief Struct to store coupled interfaces for the 2D mesh
 *
 */
template <> struct coupled_interfaces<specfem::dimension::type::dim2> {
public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  coupled_interfaces()
      : elastic_acoustic(), acoustic_poroelastic(), elastic_poroelastic() {};

  coupled_interfaces(specfem::mesh::interface_container<
                         dimension, specfem::element::medium_tag::elastic_psv,
                         specfem::element::medium_tag::acoustic>
                         elastic_acoustic,
                     specfem::mesh::interface_container<
                         dimension, specfem::element::medium_tag::acoustic,
                         specfem::element::medium_tag::poroelastic>
                         acoustic_poroelastic,
                     specfem::mesh::interface_container<
                         dimension, specfem::element::medium_tag::elastic_psv,
                         specfem::element::medium_tag::poroelastic>
                         elastic_poroelastic)
      : elastic_acoustic(elastic_acoustic),
        acoustic_poroelastic(acoustic_poroelastic),
        elastic_poroelastic(elastic_poroelastic) {};
  ///@}

  /**
   * @brief Get the interface object for the given medium types
   *
   * @tparam Medium1 Medium type 1
   * @tparam Medium2 Medium type 2
   * @return std::variant Interface object for the given medium types
   */
  template <specfem::element::medium_tag Medium1,
            specfem::element::medium_tag Medium2>
  specfem::mesh::interface_container<dimension, Medium1, Medium2> get() const;

  specfem::mesh::interface_container<dimension,
                                     specfem::element::medium_tag::elastic_psv,
                                     specfem::element::medium_tag::acoustic>
      elastic_acoustic; ///< Elastic-acoustic interfaces

  specfem::mesh::interface_container<dimension,
                                     specfem::element::medium_tag::acoustic,
                                     specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic; ///< Acoustic-poroelastic interfaces

  specfem::mesh::interface_container<dimension,
                                     specfem::element::medium_tag::elastic_psv,
                                     specfem::element::medium_tag::poroelastic>
      elastic_poroelastic; ///< Elastic-poroelastic interfaces
};

} // namespace mesh
} // namespace specfem
