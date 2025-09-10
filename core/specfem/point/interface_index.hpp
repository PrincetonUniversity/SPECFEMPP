
#pragma once

#include "edge_index.hpp"
#include "enumerations/interface.hpp"

namespace specfem::point {

/**
 * @brief Index pair for coupled interface points
 *
 * Contains edge indices for both sides of a coupled interface between
 * different physical media. Used to locate corresponding points on
 * interfaces between acoustic and elastic domains.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::dimension::type DimensionTag> class interface_index {
public:
  /** @brief Edge index on the self side of the interface */
  specfem::point::edge_index<DimensionTag> self_index;
  /** @brief Edge index on the coupled side of the interface */
  specfem::point::edge_index<DimensionTag> coupled_index;

  /** @brief Default constructor */
  KOKKOS_INLINE_FUNCTION
  interface_index() = default;

  /**
   * @brief Constructs interface index from self and coupled edge indices
   *
   * @param self_index Edge index on the self side of interface
   * @param coupled_index Edge index on the coupled side of interface
   */
  KOKKOS_INLINE_FUNCTION
  interface_index(const specfem::point::edge_index<DimensionTag> &self_index,
                  const specfem::point::edge_index<DimensionTag> &coupled_index)
      : self_index(self_index), coupled_index(coupled_index) {}
};

} // namespace specfem::point
