#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/field_derivative_storage.hpp"
#include "specfem/assembly/field_derivative_storage/impl/field_derivative_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief 3D per-GLL-point field derivative storage for attenuation field
 * derivatives bookkeeping.
 *
 * Follows the same pattern as the 2D specialization. For attenuation_none
 * combinations the sub-struct is empty (zero overhead).
 */
template <>
struct FieldDerivativeStorage<specfem::element::dimension_tag::dim3>
    : specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::field_derivatives,
          specfem::element::dimension_tag::dim3> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::field_derivatives,
      specfem::element::dimension_tag::dim3>;

  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                      DECLARE(((specfem::assembly::field_derivative_storage::
                                    impl::field_derivative_medium,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               fd_medium)))

  FieldDerivativeStorage() = default;

  FieldDerivativeStorage(
      const specfem::assembly::element_types<
          specfem::element::dimension_tag::dim3> &element_types,
      const int nspec_global, const int ngllz, const int nglly,
      const int ngllx);

  template <specfem::element::medium_tag MediumTag>
  KOKKOS_INLINE_FUNCTION constexpr specfem::assembly::field_derivative_storage::
      impl::field_derivative_medium<specfem::element::dimension_tag::dim3,
                                    MediumTag> const &
      get_container() const {

    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                        CAPTURE(fd_medium) {
                          if constexpr (_medium_tag_ == MediumTag) {
                            return _fd_medium_;
                          }
                        })
    Kokkos::abort(
        "Invalid tag combination in FieldDerivativeStorage::get_container");

    SUPPRESS_UNREACHABLE(return {};)
  }

  void copy_to_host() {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                        CAPTURE(fd_medium) { _fd_medium_.copy_to_host(); })
  }

  void copy_to_device() {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                        CAPTURE(fd_medium) { _fd_medium_.copy_to_device(); })
  }
};

} // namespace specfem::assembly
