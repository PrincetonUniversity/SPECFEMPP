#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Information about interfaces between two media
 *
 * @tparam Medium1 Medium type 1
 * @tparam Medium2 Medium type 2
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag Medium1,
          specfem::element::medium_tag Medium2>
struct interface_container;

template <specfem::element::medium_tag Medium1,
          specfem::element::medium_tag Medium2>

struct interface_container<specfem::dimension::type::dim2, Medium1, Medium2> {
  constexpr static auto dimension =
      specfem::dimension::type::dim2;          ///< Dimension
  constexpr static auto medium1_tag = Medium1; ///< Medium 1 tag
  constexpr static auto medium2_tag = Medium2; ///< Medium 2 tag

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  interface_container() {};

  interface_container(const int num_interfaces);

  ///@}

  int num_interfaces = 0; ///< Number of edges within this interface
  Kokkos::View<int *, Kokkos::HostSpace>
      medium1_index_mapping; ///< spectral element index for edges in medium 1

  Kokkos::View<specfem::mesh_entity::type *,
               Kokkos::HostSpace>
      medium1_edge_type; ///< Edge type for edges in medium 1

  Kokkos::View<int *, Kokkos::HostSpace>
      medium2_index_mapping; ///< spectral element index for edges in medium 2

  Kokkos::View<specfem::mesh_entity::type *,
               Kokkos::HostSpace>
      medium2_edge_type; ///< Edge type for edges in medium 2

  /**
   * @brief get the spectral element index for the given edge index in the given
   * medium
   *
   * @tparam medium Medium where the edge is located
   * @param interface_index Edge index
   * @return int Spectral element index
   */
  template <specfem::element::medium_tag medium>
  int get_spectral_elem_index(const int interface_index) const;

  /**
   * @brief Swapped assignment operator for interface_container.
   *
   * This operator allows assigning an interface_container with swapped medium
   * tags. It copies the number of interfaces and swaps the index mappings and
   * edge types between medium1 and medium2. This is useful for scenarios where
   * the roles of the two media need to be interchanged.
   *
   * @param other The interface_container instance with medium1 and medium2 tags
   * to swap from.
   * @return Reference to the modified interface_container with swapped media
   * data.
   */
  operator interface_container<dimension, Medium2, Medium1>() const {
    interface_container<dimension, Medium2, Medium1> other;
    other.num_interfaces = num_interfaces;
    other.medium1_index_mapping = medium2_index_mapping;
    other.medium1_edge_type = medium2_edge_type;
    other.medium2_index_mapping = medium1_index_mapping;
    other.medium2_edge_type = medium1_edge_type;
    return other;
  }
};
} // namespace mesh
} // namespace specfem
