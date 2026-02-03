#pragma once
#include "kokkos_abstractions.h"
#include "specfem/setup.hpp"

/**
 * @namespace specfem::jacobian
 * @brief The jacobian namespace contains functions for computing the Jacobian
 * matrix and global coordinates from local coordinates.
 */
namespace specfem::jacobian {} // namespace specfem::jacobian

#include "jacobian/dim2/jacobian.hpp"
#include "jacobian/dim3/jacobian.hpp"
