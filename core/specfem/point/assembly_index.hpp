#pragma once

#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Global assembly index container for quadrature points in spectral
 * element meshes.
 *
 * The assembly_index struct encapsulates the global indexing information for
 * quadrature points within the finite element assembly process. This global
 * index is essential for mapping local quadrature point data to the global
 * degrees of freedom (DOF) system, enabling efficient assembly of global
 * matrices and vectors.
 *
 * @tparam using_simd Flag controlling SIMD vectorization support.
 *                   When `true`, enables vectorized operations on multiple
 * indices. When `false`, uses scalar operations for single indices.
 *
 * @note The global index typically follows a continuous numbering scheme across
 * all elements in the mesh, often ordered to optimize memory access patterns
 * and reduce matrix bandwidth.
 *
 * @see specfem::point::index for local element indexing
 * @see specfem::data_access::Accessor for the base data access interface
 *
 * @code
 * // Example: Using assembly index in global matrix assembly
 * using AssemblyIndex = specfem::point::assembly_index<false>;
 * AssemblyIndex global_idx(1247);  // Global DOF number
 *
 * // Use in assembly operations
 * global_matrix(global_idx.iglob, global_idx.iglob) += local_contribution;
 * global_vector(global_idx.iglob) += source_term;
 * @endcode
 */
template <bool using_simd = false> struct assembly_index;

/**
 * @brief Scalar specialization for global assembly indexing of quadrature
 * points.
 *
 * This specialization handles single quadrature point global indexing without
 * SIMD vectorization. It stores the global index (iglob) that uniquely
 * identifies a quadrature point across the entire finite element mesh. This
 * index is used extensively in the assembly process to map local element
 * contributions to the correct positions in global matrices and vectors.
 *
 * The global index is typically computed during mesh preprocessing and
 * represents a continuous numbering of all quadrature points in the domain. The
 * ordering may be optimized for:
 * - Memory locality and cache efficiency
 * - Reduced matrix bandwidth in linear solvers
 * - Load balancing in parallel computations
 * - Communication patterns in distributed systems
 *
 * @note This class inherits from the SPECFEMPP data access system to provide
 * consistent interface patterns and type traits for field data management.
 *
 * @see specfem::point::assembly_index<true> for SIMD vectorized version
 * @see specfem::compute::assembly for global assembly operations
 *
 * @code
 * // Example: Global assembly workflow
 * specfem::point::assembly_index<false> global_idx;
 *
 * // Load global index from mesh data
 * specfem::assembly::load_on_device(local_index, mesh_data, global_idx);
 *
 * // Use in global matrix assembly
 * Kokkos::atomic_add(&global_stiffness(global_idx.iglob, global_idx.iglob),
 *                    element_stiffness_contribution);
 *
 * // Use in right-hand side vector assembly
 * Kokkos::atomic_add(&rhs_vector(global_idx.iglob), force_contribution);
 * @endcode
 */
template <>
struct assembly_index<false>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::assembly_index,
          specfem::dimension::type::dim2, false> {
  int iglob; ///< Global index number uniquely identifying this quadrature point
             ///< across the entire finite element mesh. Used for global DOF
             ///< mapping in assembly operations and parallel communications.

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor for uninitialized global index.
   *
   * Creates an assembly index object with uninitialized global index.
   * The iglob member should be set explicitly before use in assembly
   * operations.
   *
   * @note Marked with `KOKKOS_FUNCTION` for device/host portability.
   */
  KOKKOS_FUNCTION
  assembly_index() = default;

  /**
   * @brief Constructor with explicit global index value.
   *
   * Initializes the assembly index with a specific global index number.
   * This constructor is typically used when the global index is known
   * from mesh preprocessing or computed during element traversal.
   *
   * @param iglob Global index number for this quadrature point.
   *              Must be a valid index within [0, total_quadrature_points).
   *
   * @note The global index should be unique across the entire mesh
   *       to ensure correct assembly behavior.
   */
  KOKKOS_FUNCTION
  assembly_index(const int &iglob) : iglob(iglob) {}
  ///@}
};

/**
 * @brief SIMD vectorized specialization for global assembly indexing.
 *
 * This specialization handles vectorized assembly operations on multiple
 * quadrature points simultaneously using SIMD (Single Instruction, Multiple
 * Data) instructions. It stores global indices for a group of quadrature points
 * that can be processed together for improved computational efficiency.
 *
 * SIMD assembly indices are particularly beneficial for:
 * - Vectorized assembly operations on groups of elements
 * - Improved memory throughput through batched access patterns
 * - Enhanced performance on modern vector architectures (AVX, NEON)
 * - Reduced loop overhead in computationally intensive kernels
 *
 * The SIMD index contains a base global index and tracks how many valid points
 * are present in the current SIMD lane, enabling masked operations for
 * irregular element groups at domain boundaries.
 *
 * @note This specialization is designed for use with SIMD-enabled field
 * operations and requires careful attention to lane masking for correct
 * results.
 *
 * @see specfem::point::assembly_index<false> for scalar version
 * @see specfem::datatype::simd for SIMD type definitions
 *
 * @code
 * // Example: SIMD assembly operations
 * using SIMDAssemblyIndex = specfem::point::assembly_index<true>;
 * SIMDAssemblyIndex simd_idx(base_global_index, 4);  // 4 points in vector
 *
 * // Process vectorized assembly with lane masking
 * for (int lane = 0; lane < simd_idx.number_points; ++lane) {
 *   if (simd_idx.mask(lane)) {
 *     // Valid lane: perform assembly operation
 *     global_vector(simd_idx.iglob + lane) += simd_contributions[lane];
 *   }
 * }
 * @endcode
 */
template <>
struct assembly_index<true>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::assembly_index,
          specfem::dimension::type::dim2, true> {
  int number_points; ///< Number of points in the SIMD vector
  int iglob;         ///< Global index number of the quadrature point

  /**
   * @brief Mask function to determine if a lane is valid
   *
   * @param lane Lane index
   */
  KOKKOS_FUNCTION
  bool mask(const std::size_t &lane) const { return int(lane) < number_points; }

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  assembly_index() = default;

  /**
   * @brief Constructor with values
   *
   * @param iglob Global index number of the quadrature point
   * @param number_points Number of points in the SIMD vector
   */
  KOKKOS_FUNCTION
  assembly_index(const int &iglob, const int &number_points)
      : number_points(number_points), iglob(iglob) {}
  ///@}
};

/**
 * @brief Type alias for the SIMD assembly index
 *
 */
using simd_assembly_index = assembly_index<true>;

} // namespace point
} // namespace specfem
