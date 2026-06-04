#pragma once

#include "specfem/data_access/container.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly::field_derivative_storage::impl {

/**
 * @brief Primary template for field derivative storage at a set of GLL points.
 *
 * Stores the du (strain) tensor per GLL point for elements belonging to a given
 * DimensionTag, MediumTag combination. Used to record
 * the field derivatives from the previous time step for attenuation strain
 * calculations.
 *
 * Storage is compact: only elements with this specific tag combination are
 * stored, indexed via a global ispec → compact index mapping.
 *
 * This primary template covers all AttenuationTag values except
 * attenuation_none. For attenuation_none, an empty specialization below
 * provides zero overhead.
 *
 * @tparam DimensionTag       Dimension of the elements
 * @tparam MediumTag    Medium of the elements
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct field_derivative_medium
    : specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::field_derivatives,
          DimensionTag> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::field_derivatives, DimensionTag>;

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;

  /**
   * @name Compile-time constants
   */
  ///@{
  static constexpr int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  static constexpr int num_dimensions =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  ///@}

  using view_type = typename base_type::template tensor_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /// Integer index view for the compact → global ispec mapping
  using index_view_type = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  /// Compact storage: shape
  /// [nspec_attn][ngllz][...][components][num_dimensions]
  view_type du;
  typename view_type::host_mirror_type h_du;

  /// Maps global ispec → compact index (or -1 for non-matching elements)
  index_view_type ispec_to_compact;
  typename index_view_type::host_mirror_type h_ispec_to_compact;

  field_derivative_medium() = default;

  /**
   * @brief Construct storage for the given element list.
   *
   * @param elements     Host view of global ispec indices belonging to this
   *                     medium
   * @param nspec_global Total number of spectral elements in the mesh (for
   *                     the inverse mapping view).
   * @param ngllz        Number of GLL points in z-direction.
   * @param ngllx        Number of GLL points in x-direction.
   */
  template <
      specfem::element::dimension_tag D = DimensionTag,
      std::enable_if_t<D == specfem::element::dimension_tag::dim2, int> = 0>
  field_derivative_medium(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> &elements,
      const int nspec_global, const int ngllz, const int ngllx) {

    const int nspec_attn = static_cast<int>(elements.extent(0));

    du = view_type("field_derivative_storage", nspec_attn, ngllz, ngllx,
                   components, num_dimensions);
    h_du = typename view_type::host_mirror_type("h_field_derivative_storage",
                                                nspec_attn, ngllz, ngllx,
                                                components, num_dimensions);
    Kokkos::deep_copy(du, static_cast<type_real>(0));

    // Build ispec → compact-index inverse mapping (initialized to -1)
    ispec_to_compact =
        index_view_type("ispec_to_compact_field_deriv", nspec_global);
    h_ispec_to_compact = Kokkos::create_mirror_view(ispec_to_compact);
    Kokkos::deep_copy(h_ispec_to_compact, -1);

    for (int i = 0; i < nspec_attn; ++i) {
      h_ispec_to_compact(elements(i)) = i;
    }

    copy_to_host();   // sync h_du from zeroed device view
    copy_to_device(); // push inverse mapping to device
  }

  /**
   * @brief Construct storage for the given element list.
   *
   * @param elements     Host view of global ispec indices belonging to this
   *                     medium.
   * @param nspec_global Total number of spectral elements in the mesh (for
   *                     the inverse mapping view).
   * @param ngllz        Number of GLL points in z-direction.
   * @param nglly        Number of GLL points in y-direction
   * @param ngllx        Number of GLL points in x-direction.
   */
  template <
      specfem::element::dimension_tag D = DimensionTag,
      std::enable_if_t<D == specfem::element::dimension_tag::dim3, int> = 0>
  field_derivative_medium(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> &elements,
      const int nspec_global, const int ngllz, const int nglly,
      const int ngllx) {

    const int nspec_attn = static_cast<int>(elements.extent(0));

    du = view_type("field_derivative_storage", nspec_attn, ngllz, nglly, ngllx,
                   components, num_dimensions);
    h_du = typename view_type::host_mirror_type("h_field_derivative_storage",
                                                nspec_attn, ngllz, nglly, ngllx,
                                                components, num_dimensions);

    Kokkos::deep_copy(du, static_cast<type_real>(0));

    // Build ispec → compact-index inverse mapping (initialized to -1)
    ispec_to_compact =
        index_view_type("ispec_to_compact_field_deriv", nspec_global);
    h_ispec_to_compact = Kokkos::create_mirror_view(ispec_to_compact);
    Kokkos::deep_copy(h_ispec_to_compact, -1);
    for (int i = 0; i < nspec_attn; ++i) {
      h_ispec_to_compact(elements(i)) = i;
    }

    copy_to_host();   // sync h_du from zeroed device view
    copy_to_device(); // push inverse mapping to device
  }

  void copy_to_host() {
    specfem::datatype::deep_copy(h_du, du);
    Kokkos::deep_copy(h_ispec_to_compact, ispec_to_compact);
  }

  void copy_to_device() {
    specfem::datatype::deep_copy(du, h_du);
    Kokkos::deep_copy(ispec_to_compact, h_ispec_to_compact);
  }

  /**
   * @brief Load field derivatives for a non-SIMD GLL point from compact
   * storage.
   */
  template <typename PointFDType, typename IndexType,
            std::enable_if_t<!IndexType::using_simd, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_values(const IndexType &index, PointFDType &point_fd) const {
    const int i = ispec_to_compact(index.ispec);
    for (int ic = 0; ic < PointFDType::components; ++ic) {
      for (int id = 0; id < PointFDType::num_dimensions; ++id) {
        if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
          point_fd.du[ic][id] = du(i, index.iz, index.ix, ic, id);
        } else {
          point_fd.du[ic][id] = du(i, index.iz, index.iy, index.ix, ic, id);
        }
      }
    }
  }

  /**
   * @brief Load field derivatives for a SIMD GLL point from compact storage.
   *
   * Uses copy_from to load SIMD_WIDTH consecutive compact-index values
   * starting at the compact index of the first lane, matching the
   * jacobian_matrix SIMD pattern.
   */
  template <typename PointFDType, typename IndexType,
            std::enable_if_t<IndexType::using_simd, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_values(const IndexType &index, PointFDType &point_fd) const {
    using simd_type = typename PointFDType::simd;
    using mask_type = typename simd_type::mask_type;
    using tag_type = typename simd_type::tag_type;

    const int i = ispec_to_compact(index.ispec);
    const auto mask = index.template get_mask<simd_type>();

    for (int ic = 0; ic < PointFDType::components; ++ic) {
      for (int id = 0; id < PointFDType::num_dimensions; ++id) {
        if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
          point_fd.du[ic][id] = Kokkos::Experimental::simd_partial_load(
              &du(i, index.iz, index.ix, ic, id), mask, tag_type());
        } else {
          point_fd.du[ic][id] = Kokkos::Experimental::simd_partial_load(
              &du(i, index.iz, index.iy, index.ix, ic, id), mask, tag_type());
        }
      }
    }
  }

  /**
   * @brief Store field derivatives for a non-SIMD GLL point into compact
   * storage.
   */
  template <typename PointFDType, typename IndexType,
            std::enable_if_t<!IndexType::using_simd, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointFDType &point_fd) {
    const int i = ispec_to_compact(index.ispec);
    for (int ic = 0; ic < PointFDType::components; ++ic) {
      for (int id = 0; id < PointFDType::num_dimensions; ++id) {
        if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
          du(i, index.iz, index.ix, ic, id) = point_fd.du[ic][id];
        } else {
          du(i, index.iz, index.iy, index.ix, ic, id) = point_fd.du[ic][id];
        }
      }
    }
  }

  /**
   * @brief Store field derivatives for a SIMD GLL point into compact storage
   * (dim2).
   *
   * Uses simd_partial_store to scatter SIMD_WIDTH consecutive compact-index
   * values starting at the compact index of the first lane, matching the
   * jacobian_matrix SIMD pattern.
   */
  template <typename PointFDType, typename IndexType,
            std::enable_if_t<(IndexType::using_simd &&
                              IndexType::dimension_tag ==
                                  specfem::element::dimension_tag::dim2),
                             int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointFDType &point_fd) {
    using simd_type = typename PointFDType::simd;
    using mask_type = typename simd_type::mask_type;
    using tag_type = typename simd_type::tag_type;

    const int i = ispec_to_compact(index.ispec);
    const auto mask = index.template get_mask<simd_type>();

    for (int ic = 0; ic < PointFDType::components; ++ic) {
      for (int id = 0; id < PointFDType::num_dimensions; ++id) {
        Kokkos::Experimental::simd_partial_store(
            point_fd.du[ic][id], &du(i, index.iz, index.ix, ic, id), mask,
            tag_type());
      }
    }
  }

  /**
   * @brief Store field derivatives for a SIMD GLL point into compact storage
   * (dim3).
   *
   * Uses simd_partial_store to scatter SIMD_WIDTH consecutive compact-index
   * values starting at the compact index of the first lane. The du view uses
   * Kokkos::LayoutLeft so consecutive compact indices for a fixed (iz,iy,ix)
   * are contiguous in memory (stride-1 along the first axis).
   */
  template <typename PointFDType, typename IndexType,
            std::enable_if_t<(IndexType::using_simd &&
                              IndexType::dimension_tag ==
                                  specfem::element::dimension_tag::dim3),
                             int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointFDType &point_fd) {
    using simd_type = typename PointFDType::simd;
    using mask_type = typename simd_type::mask_type;
    using tag_type = typename simd_type::tag_type;

    const int i = ispec_to_compact(index.ispec);
    const auto mask = index.template get_mask<simd_type>();

    for (int ic = 0; ic < PointFDType::components; ++ic) {
      for (int id = 0; id < PointFDType::num_dimensions; ++id) {
        Kokkos::Experimental::simd_partial_store(
            point_fd.du[ic][id], &du(i, index.iz, index.iy, index.ix, ic, id),
            mask, tag_type());
      }
    }
  }
};

} // namespace specfem::assembly::field_derivative_storage::impl
