#pragma once

namespace specfem::element_connections {

/**
 * @brief Connection conformity types between mesh elements.
 */
enum class type : int {
  strongly_conforming = 1, ///< Nodes match exactly
  weakly_conforming = 2, ///< Nodes match, shape functions may be discontinuous
  nonconforming = 3      ///< No matching nodes, geometrically adjacent
};

} // namespace specfem::element_connections
