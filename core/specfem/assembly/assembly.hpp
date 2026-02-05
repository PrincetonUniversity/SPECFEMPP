#pragma once

#include "enumerations/interface.hpp"

/**
 * @namespace specfem::assembly
 * @brief Data structures for SEM computational data
 *
 * Provides core infrastructure for SEM simulations, managing computational data
 * for finite element kernels. Transforms mesher-supplied element data and
 * computes per-GLL-point values.
 *
 * **Organization:**
 *
 * - Containers : The namespace contains data storage containers (@ref
 * specfem::assembly::mesh, @ref specfem::assembly::jacobian_matrix, etc) that
 * store data computed at all GLL points
 * - Data access functions : Each container provides functions to load/store
 * data on device/host (e.g., @c load_on_device , @c store_on_device )
 *
 */
namespace specfem::assembly {

/**
 * @brief Data class used to store computational data required for SEM
 * simulations
 *
 * Provides classes to transform mesher-supplied element data and compute
 * per-GLL-point values. The per-GLL-point data is stored in @c Kokkos::Views
 * which provide portability & data management across CPU and GPU architectures.
 * The assembly class is specialized for 2D and 3D
 * problems.
 */
template <specfem::element::dimension_tag DimensionTag> struct assembly;

} // namespace specfem::assembly

#include "assembly/dim2/assembly.hpp"
#include "assembly/dim3/assembly.hpp"
