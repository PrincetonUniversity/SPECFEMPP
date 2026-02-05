#pragma once

/**
 * @brief Chunk-based data accessors for face operations in 3D spectral elements
 *
 * Provides vectorized access to field data and interface coupling terms
 * for chunks of element faces in the 3D spectral element method.
 *
 * Key components:
 * - Field accessors (displacement, velocity, acceleration)
 * - Index mapping for chunk operations
 *
 * Faces are 2D surfaces in 3D elements, with ngll x ngll quadrature points.
 */
namespace specfem::chunk_face {}

#include "chunk_face/acceleration.hpp"
#include "chunk_face/displacement.hpp"
#include "chunk_face/index.hpp"
#include "chunk_face/velocity.hpp"
