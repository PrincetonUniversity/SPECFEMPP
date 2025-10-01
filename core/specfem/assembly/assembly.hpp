#pragma once

#include "enumerations/interface.hpp"

/**
 * @brief Assembly namespace defines data structures used to store data related
 * to finite element assembly.
 *
 * The data is organized in a manner that makes it effiecient to access when
 * computing finite element compute kernels.
 */
namespace specfem::assembly {

/**
 * @brief Assembly class for finite element problems in different dimensions.
 *
 * @tparam DimensionTag
 */
template <specfem::dimension::type DimensionTag> struct assembly;

} // namespace specfem::assembly

#include "assembly/dim2/assembly.hpp"
#include "assembly/dim3/assembly.hpp"
