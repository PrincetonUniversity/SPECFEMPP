#pragma once

#include "enumerations/medium.hpp"
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
template <specfem::dimension::type DimensionType,
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
  interface_container(){};

  interface_container(const int num_interfaces);

  ///@}

  int num_interfaces = 0; ///< Number of edges within this interface
  Kokkos::View<int *, Kokkos::HostSpace>
      medium1_index_mapping; ///< spectral element index for edges in medium 1

  Kokkos::View<int *, Kokkos::HostSpace>
      medium2_index_mapping; ///< spectral element index for edges in medium 2

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
};

// template<specfem::element::medium_tag Medium1,
//          specfem::element::medium_tag Medium2>
// struct interface_container<specfem::dimension::type::dim3, Medium1, Medium2>
// {

// };
} // namespace mesh
} // namespace specfem
