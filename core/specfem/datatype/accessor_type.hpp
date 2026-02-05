#pragma once

namespace specfem::datatype {

/**
 * @brief Data access patterns for spectral element simulations.
 */
enum class AccessorType {
  point,         ///< Single quadrature point access
  element,       ///< Full element access
  chunk_element, ///< Chunked element access for vectorization
  chunk_edge,    ///< Chunked edge access for interfaces (2D)
  chunk_face     ///< Chunked face access for interfaces (3D)
};

} // namespace specfem::datatype
