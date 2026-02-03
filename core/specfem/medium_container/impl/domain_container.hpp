#pragma once

#include "domain_view.hpp"
#include "specfem/element.hpp"
#include "specfem/macros.hpp"
#include <boost/preprocessor.hpp>

namespace specfem::medium_container::impl {

// Helper function to get flat index from mapping for dim2
template <typename MappingType, typename IndexType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<IndexType::dimension_tag == specfem::dimension::type::dim2,
                     std::size_t>
    get_flat_index(const MappingType &mapping, const IndexType &index) {
  return mapping(index.ispec, index.iz, index.ix);
}

// Helper function to get flat index from mapping for dim3
template <typename MappingType, typename IndexType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<IndexType::dimension_tag == specfem::dimension::type::dim3,
                     std::size_t>
    get_flat_index(const MappingType &mapping, const IndexType &index) {
  return mapping(index.ispec, index.iz, index.iy, index.ix);
}

} // namespace specfem::medium_container::impl

namespace specfem::medium_container::properties {

/**
 * @brief Material properties storage container template.
 *
 * Base data container for storing material properties (density, elastic moduli,
 * wave speeds, etc.) at quadrature points within spectral elements. Uses the
 * DATA_CONTAINER macro to generate efficient Kokkos-based storage with
 * device/host synchronization capabilities.
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename Enable = void>
struct data_container;

} // namespace specfem::medium_container::properties

namespace specfem::medium_container::kernels {

/**
 * @brief Sensitivity kernel storage container template.
 *
 * Base data container for storing misfit kernels (Frechet derivatives)
 * representing gradients of the misfit function with respect to material
 * parameters and are accumulated during adjoint simulations from
 * forward/adjoint wavefield correlations.
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename Enable = void>
struct data_container;
} // namespace specfem::medium_container::kernels
