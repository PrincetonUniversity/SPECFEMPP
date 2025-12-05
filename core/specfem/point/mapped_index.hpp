#pragma once

#include "enumerations/interface.hpp"
#include "index.hpp"
#include "specfem/data_access.hpp"

namespace specfem {
namespace point {

/**
 * @brief Extended indexing system for quadrature points with parameter mapping
 * support.
 *
 * The mapped_index class extends the basic quadrature point indexing system by
 * adding an additional mapping index (imap) that associates external data
 * arrays or parameters with specific quadrature points. This enables element
 * iterators with multiple points inside an element.
 *
 * @tparam DimensionTag Spatial dimension of the element indexing system.
 *                      Determines the underlying index structure (2D or 3D).
 * @tparam UseSIMD Boolean flag controlling SIMD vectorization support.
 *                 When enabled, allows vectorized operations on index arrays.
 *
 * @note The mapping index (imap) should be validated against the size of
 *       associated data arrays to prevent out-of-bounds access errors.
 *
 * @see specfem::point::index for base quadrature point indexing
 * @see specfem::compute::properties for material property management
 *
 * @code
 * // Example: Material property mapping for heterogeneous media
 * using MaterialMappedIndex = specfem::point::mapped_index<
 *     specfem::dimension::type::dim2, false>;
 *
 * // Create array of material parameters
 * Kokkos::View<type_real*> material_density("density", num_materials);
 * material_density(0) = 1000.0;  // Water
 * material_density(1) = 2700.0;  // Rock
 * material_density(2) = 7800.0;  // Steel
 *
 * // Create mapped index linking quadrature point to material
 * specfem::point::index<specfem::dimension::type::dim2, false> quad_pt(elem,
 * iz, ix); MaterialMappedIndex mapped_pt(quad_pt, material_id);
 *
 * // Access material property through mapping
 * type_real local_density = material_density(mapped_pt.imap);
 *
 * // Use in assembly operations
 * mass_matrix_contrib *= local_density * quadrature_weight;
 * @endcode
 *
 * @code
 * // Example: Source term distribution mapping
 * Kokkos::View<type_real**> source_amplitudes("sources", num_sources,
 * num_components);
 *
 * // Map quadrature point to specific source in array
 * MaterialMappedIndex source_pt(quad_index, source_id);
 *
 * // Apply source term with mapped amplitude
 * for (int icomp = 0; icomp < num_components; ++icomp) {
 *   type_real source_value = source_amplitudes(source_pt.imap, icomp);
 *   rhs_vector(iglob) += source_value * basis_function * time_function;
 * }
 * @endcode
 */
template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct mapped_index : public index<DimensionTag, UseSIMD> {
private:
  using base_type = index<DimensionTag, UseSIMD>;

public:
  int imap; ///< Mapping index for indirect data array access. Points to an
            ///< entry in associated parameter arrays (material properties,
            ///< sources, boundary data, etc.). Must be within [0, array_size)
            ///< for valid access.

  /**
   * @brief Constructor creating mapped index from base index and mapping value.
   *
   * Constructs a mapped index that combines quadrature point location
   * information with an additional mapping index for parameter lookup. This
   * enables efficient association of external data with specific spatial
   * locations in the mesh.
   *
   * @param index Base quadrature point index containing element and local
   * coordinates. Provides spatial location within the spectral element mesh.
   * @param imap Mapping index for parameter array lookup. Should correspond to
   *             a valid entry in the associated data structure (e.g., material
   * ID, source index, boundary condition index).
   *
   * @note The caller is responsible for ensuring imap is within valid bounds
   *       for the intended parameter arrays to prevent access violations.
   *
   * @see specfem::point::index for base indexing functionality
   */
  KOKKOS_INLINE_FUNCTION
  mapped_index(const base_type &index, const int &imap)
      : base_type(index), imap(imap) {}
};

} // namespace point
} // namespace specfem
