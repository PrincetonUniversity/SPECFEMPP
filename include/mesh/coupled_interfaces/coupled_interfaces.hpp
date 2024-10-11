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
struct coupled_interfaces {
public:
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
  /**
   * @brief Constructor to read and assign values from fortran binary database
   * file
   *
   * @param stream Stream object for fortran binary file buffered to coupled
   * interfaces section
   * @param num_interfaces_elastic_acoustic Number of elastic-acoustic
   * interfaces
   * @param num_interfaces_acoustic_poroelastic Number of acoustic-poroelastic
   * interfaces
   * @param num_interfaces_elastic_poroelastic Number of elastic-poroelastic
   * interfaces
   * @param mpi Pointer to MPI object
   */
  coupled_interfaces(std::ifstream &stream,
                     const int num_interfaces_elastic_acoustic,
                     const int num_interfaces_acoustic_poroelastic,
                     const int num_interfaces_elastic_poroelastic,
                     const specfem::MPI::MPI *mpi);
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
                   specfem::element::medium_tag::elastic,
                   specfem::element::medium_tag::acoustic>,
               specfem::mesh::interface_container<
                   specfem::element::medium_tag::acoustic,
                   specfem::element::medium_tag::poroelastic>,
               specfem::mesh::interface_container<
                   specfem::element::medium_tag::elastic,
                   specfem::element::medium_tag::poroelastic> >
  get() const;

  specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                     specfem::element::medium_tag::acoustic>
      elastic_acoustic; ///< Elastic-acoustic interfaces

  specfem::mesh::interface_container<specfem::element::medium_tag::acoustic,
                                     specfem::element::medium_tag::poroelastic>
      acoustic_poroelastic; ///< Acoustic-poroelastic interfaces

  specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                     specfem::element::medium_tag::poroelastic>
      elastic_poroelastic; ///< Elastic-poroelastic interfaces
};

} // namespace mesh
} // namespace specfem
