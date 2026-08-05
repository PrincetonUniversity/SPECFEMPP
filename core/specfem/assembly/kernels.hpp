#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/impl/domain_kernels.hpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/enums.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Data container for misfit kernels within the spectral element domain
 *
 * This class provides storage and data access functions for the misfit kernels
 * associated with spectral elements in spectral element mesh.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag>
struct kernels
    : public impl::value_containers<DimensionTag, impl::domain_kernels> {
public:
  kernels() = default;
  kernels(const specfem::assembly::element_types<DimensionTag> &element_types)
      : impl::value_containers<DimensionTag, impl::domain_kernels>(
            element_types.nspec, element_types.element_grid,
            [&element_types]<typename TagsType>() {
              return specfem::assembly::impl::domain_kernels<
                  TagsType::dimension_tag, TagsType::medium_tag,
                  TagsType::property_tag>(
                  element_types.get_elements_on_host(TagsType::medium_tag,
                                                     TagsType::property_tag),
                  element_types.element_grid);
            }) {}
};

} // namespace specfem::assembly

extern template struct specfem::assembly::kernels<
    specfem::element::dimension_tag::dim2>;
extern template struct specfem::assembly::kernels<
    specfem::element::dimension_tag::dim3>;

// Device-side data access functions for GPU kernels
#include "kernels/data_access/add_on_device.hpp" ///< Accumulate kernel data on GPU device
#include "kernels/data_access/load_on_device.hpp" ///< Load kernel data on GPU device
#include "kernels/data_access/store_on_device.hpp" ///< Store kernel data on GPU device

// Host-side data access functions for CPU operations
#include "kernels/data_access/add_on_host.hpp" ///< Accumulate kernel data on CPU host
#include "kernels/data_access/load_on_host.hpp" ///< Load kernel data on CPU host
#include "kernels/data_access/store_on_host.hpp" ///< Store kernel data on CPU host
