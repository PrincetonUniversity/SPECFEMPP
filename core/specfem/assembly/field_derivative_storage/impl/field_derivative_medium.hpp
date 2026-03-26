#pragma once

#include "specfem/data_access/container.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

/**
 * @brief Primary template for field derivative storage at a set of GLL points.
 *
 * Stores the du (strain) tensor per GLL point for elements belonging to a given
 * (DimTag, MediumTag, PropertyTag, AttenuationTag) combination. Used to record
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
 * @tparam DimTag       Dimension of the elements
 * @tparam MediumTag    Medium of the elements
 * @tparam PropertyTag  Property of the elements
 * @tparam AttenuationTag Attenuation type (must not be attenuation_none)
 */
template <specfem::element::dimension_tag DimTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag>
struct field_derivative_medium
    : specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::strain, DimTag> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::strain, DimTag>;

  /// Tensor storage view type (rank 5 for dim2, rank 6 for dim3)
  using view_type = typename base_type::template tensor_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /// Integer index view for the compact → global ispec mapping
  using index_view_type = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  /**
   * @name Compile-time constants
   */
  ///@{
  static constexpr int components =
      specfem::element::attributes<DimTag, MediumTag>::components;
  static constexpr int num_dimensions =
      specfem::element::attributes<DimTag, MediumTag>::dimension;
  ///@}

  /// Compact storage: shape
  /// [nspec_attn][ngllz][...][components][num_dimensions]
  view_type du_storage;
  typename view_type::HostMirror h_du_storage;

  /// Maps global ispec → compact index (or -1 for non-matching elements)
  index_view_type ispec_to_compact;
  typename index_view_type::HostMirror h_ispec_to_compact;

  field_derivative_medium() = default;

  /**
   * @brief Construct storage for the given element list.
   *
   * @param elements     Host view of global ispec indices belonging to this
   *                     (medium, property, attenuation) combination.
   * @param nspec_global Total number of spectral elements in the mesh (for
   *                     the inverse mapping view).
   * @param ngllz        Number of GLL points in z-direction.
   * @param nglly        Number of GLL points in y-direction (dim3 only; pass 0
   *                     or any value for dim2 — it is not used).
   * @param ngllx        Number of GLL points in x-direction.
   */
  field_derivative_medium(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> &elements,
      const int nspec_global, const int ngllz, const int nglly,
      const int ngllx) {

    const int nspec_attn = static_cast<int>(elements.extent(0));

    // Allocate compact du_storage — shape differs between dim2 and dim3
    if constexpr (DimTag == specfem::element::dimension_tag::dim2) {
      du_storage = view_type("field_derivative_storage", nspec_attn, ngllz,
                             ngllx, components, num_dimensions);
    } else {
      du_storage = view_type("field_derivative_storage", nspec_attn, ngllz,
                             nglly, ngllx, components, num_dimensions);
    }
    h_du_storage = Kokkos::create_mirror_view(du_storage);
    Kokkos::deep_copy(du_storage, static_cast<type_real>(0));

    // Build ispec → compact-index inverse mapping (initialized to -1)
    ispec_to_compact =
        index_view_type("ispec_to_compact_field_deriv", nspec_global);
    h_ispec_to_compact = Kokkos::create_mirror_view(ispec_to_compact);
    Kokkos::deep_copy(h_ispec_to_compact, -1);
    for (int i = 0; i < nspec_attn; ++i) {
      h_ispec_to_compact(elements(i)) = i;
    }

    copy_to_host();   // sync h_du_storage from zeroed device view
    copy_to_device(); // push inverse mapping to device
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_du_storage, du_storage);
    Kokkos::deep_copy(h_ispec_to_compact, ispec_to_compact);
  }

  void copy_to_device() {
    Kokkos::deep_copy(du_storage, h_du_storage);
    Kokkos::deep_copy(ispec_to_compact, h_ispec_to_compact);
  }
};

/**
 * @brief Empty specialization for attenuation_none — zero overhead.
 *
 * No views are allocated. All operations are no-ops. The compiler can
 * eliminate load/store calls to this type entirely.
 */
template <specfem::element::dimension_tag DimTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct field_derivative_medium<DimTag, MediumTag, PropertyTag,
                               specfem::element::attenuation_tag::none> {
  field_derivative_medium() = default;
  void copy_to_host() {}
  void copy_to_device() {}
};

} // namespace specfem::assembly::impl
