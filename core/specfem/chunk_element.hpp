#pragma once

/**
 * @brief Chunk-based field accessors for vectorized spectral element
 * operations.
 *
 * Provides high-performance, vectorized data structures for processing spectral
 * element fields (displacement, velocity, acceleration) and indices in chunks.
 * Optimized for SIMD operations, cache efficiency, and parallel execution on
 * modern architectures.
 */
namespace specfem::chunk_element {}

#include "chunk_element/acceleration.hpp"
#include "chunk_element/displacement.hpp"
#include "chunk_element/field_pack.hpp"
#include "chunk_element/index.hpp"
#include "chunk_element/mapped_index.hpp"
#include "chunk_element/stress_integrand.hpp"
#include "chunk_element/velocity.hpp"
