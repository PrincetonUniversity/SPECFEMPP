#pragma once

#include "enumerations/medium.hpp"
#include "interface_container.hpp"
#include "specfem_mpi/specfem_mpi.hpp"
#include <fstream>
#include <variant>

namespace specfem {
namespace mesh {
/**
 * @brief Information about coupled interfaces
 *
 */
template <typename specfem::dimension::type DimensionType>
struct coupled_interfaces {
public:
  constexpr static auto dimension = DimensionType; ///< Dimension
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
      : elastic_acoustic(), acoustic_poroelastic(), elastic_poroelastic(){};

  coupled_interfaces(specfem::mesh::interface_container<
                         DimensionType, specfem::element::medium_tag::elastic,
                         specfem::element::medium_tag::acoustic>
                         elastic_acoustic,
                     specfem::mesh::interface_container<
                         DimensionType, specfem::element::medium_tag::acoustic,
                         specfem::element::medium_tag::poroelastic>
                         acoustic_poroelastic,
                     specfem::mesh::interface_container<
                         DimensionType, specfem::element::medium_tag::elastic,
                         specfem::element::medium_tag::poroelastic>
                         elastic_poroelastic)
      : elastic_acoustic(elastic_acoustic),
        acoustic_poroelastic(acoustic_poroelastic),
        elastic_poroelastic(elastic_poroelastic){};
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
  std::variant<specfem::mesh::interface_container<
                   DimensionType, specfem::element::medium_tag::elastic,
                   specfem::element::medium_tag::acoustic>,
               specfem::mesh::interface_container<
                   DimensionType, specfem::element::medium_tag::acoustic,
                   specfem::element::medium_tag::poroelastic>,
               specfem::mesh::interface_container<
                   DimensionType, specfem::element::medium_tag::elastic,
                   specfem::element::medium_tag::poroelastic> >
  get() const;

  specfem::mesh::interface_container<DimensionType,
                                     specfem::element::medium_tag::elastic,
                                     specfem::element::medium_tag::acoustic>
      elastic_acoustic; ///< Elastic-acoustic interfaces

  specfem::mesh::interface_container<DimensionType,
                                     specfem::element::medium_tag::acoustic,
                                     specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic; ///< Acoustic-poroelastic interfaces

  specfem::mesh::interface_container<DimensionType,
                                     specfem::element::medium_tag::elastic,
                                     specfem::element::medium_tag::poroelastic>
      elastic_poroelastic; ///< Elastic-poroelastic interfaces
};

} // namespace mesh
} // namespace specfem
