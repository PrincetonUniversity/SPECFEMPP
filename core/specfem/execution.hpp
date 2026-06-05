#pragma once

/**
 * @brief Parallel execution abstractions for spectral element computations.
 *
 * Provides Kokkos-based parallel execution policies, chunked domain iterators,
 * and loop abstractions. Supports hierarchical parallelism with team-based and
 * range-based execution patterns.
 *
 * @code
 * // Parallel loop over domain elements
 * specfem::execution::for_all(domain_iterator, [=](auto index) {
 *   // Process quadrature points
 * });
 * @endcode
 *
 * @ingroup Execution
 */
namespace specfem::execution {}

#include "execution/chunked_domain_iterator.hpp"
#include "execution/chunked_edge_iterator.hpp"
#include "execution/chunked_face_iterator.hpp"
#include "execution/chunked_intersection_iterator.hpp"
#include "execution/for_all.hpp"
#include "execution/for_each_level.hpp"
#include "execution/mapped_chunked_domain_iterator.hpp"
#include "execution/policy.hpp"
#include "execution/prefetch_ahead.hpp"
#include "execution/range_iterator.hpp"
#include "execution/team_thread_md_range_iterator.hpp"
#include "execution/void_iterator.hpp"
