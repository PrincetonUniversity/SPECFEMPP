#pragma once

/**
 * @file interface.hpp
 * @brief Main interface header that includes all algorithm headers
 *
 * This header provides a convenient way to include all algorithms functionality
 * by including the individual algorithm headers.
 */

/**
 * @brief Algorithms for spectral element computations
 *
 * The algorithms namespace contains core numerical algorithms used in
 * SPECFEM++, including:
 * - Divergence and gradient computations
 * - Interpolation and point location
 * - Data transfer between mesh entities
 * - Integration and coupling
 */
namespace specfem::algorithms {}

#include "algorithms/coupling_integral.hpp"
#include "algorithms/divergence.hpp"
#include "algorithms/gradient.hpp"
#include "algorithms/interpolate.hpp"
#include "algorithms/locate_point.hpp"
#include "algorithms/transfer.hpp"
