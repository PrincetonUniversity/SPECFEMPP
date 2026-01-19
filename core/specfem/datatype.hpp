#pragma once

/**
 * @brief Primitive datatypes used within SPECFEM++. These include
 * point-based, element-based, chunk-element-based, and chunk-edge-based views
 * for scalar, vector, and tensor data. The datatypes are optimized for SIMD
 * operations and can be configured to use SIMD types or standard types based
 * on compile-time flags.
 *
 * These datatypes are generally used to build higher-level data structures
 * such as accessors that provide convenient interfaces for accessing and
 * manipulating data stored in these primitive datatypes.
 */
namespace specfem::datatype {}

#include "datatype/chunk_edge_view.hpp"
#include "datatype/chunk_element_view.hpp"
#include "datatype/element_view.hpp"
#include "datatype/point_view.hpp"
#include "datatype/simd.hpp"
