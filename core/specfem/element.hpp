#pragma once

/**
 * @brief Element configuration and utilities for spectral element
 * computations.
 *
 * Provides compile-time element configuration through template specializations
 * and tag-based type system. Includes dimension traits, physics medium types,
 * material properties, boundary conditions, and element attributes.
 *
 */
namespace specfem::element {}

#include "element/attributes.hpp"
#include "element/boundary.hpp"
#include "element/dimension.hpp"
#include "element/tags.hpp"
#include "element/to_string.hpp"
