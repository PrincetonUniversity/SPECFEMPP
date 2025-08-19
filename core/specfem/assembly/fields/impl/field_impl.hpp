#pragma once

#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

namespace fields_impl {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass>
class base_field {
private:
  int nglob;

public:
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  constexpr static auto data_class = DataClass;
  base_field() = default;
  base_field(const int nglob, std::string name)
      : nglob(nglob), data(name, nglob, components),
        h_data(Kokkos::create_mirror_view(data)) {}

  template <bool on_device, specfem::data_access::DataClassType U,
            typename std::enable_if_t<U == data_class, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &get_value(
      const std::integral_constant<specfem::data_access::DataClassType, U>,
      const int &iglob, const int &icomp) const {

    if constexpr (on_device) {
      return data(iglob, icomp);
    } else {
      return h_data(iglob, icomp);
    }
  }

  template <specfem::sync::kind SyncType> void sync() const {
    if constexpr (SyncType == specfem::sync::kind::HostToDevice) {
      Kokkos::deep_copy(data, h_data);
    } else if constexpr (SyncType == specfem::sync::kind::DeviceToHost) {
      Kokkos::deep_copy(h_data, data);
    }
  }

private:
  using ViewType = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;
  ViewType data;
  ViewType::HostMirror h_data;

protected:
  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION
      std::conditional_t<on_device, ViewType, ViewType::HostMirror>
      get_base_field_view() const {
    if constexpr (on_device) {
      return data;
    } else {
      return h_data;
    }
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class field_impl
    : public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::displacement>,
      public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::velocity>,
      public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::acceleration>,
      public base_field<DimensionTag, MediumTag,
                        specfem::data_access::DataClassType::mass_matrix> {
private:
  using displacement_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::displacement>;
  using velocity_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::velocity>;
  using acceleration_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::acceleration>;
  using mass_inverse_base_type =
      base_field<DimensionTag, MediumTag,
                 specfem::data_access::DataClassType::mass_matrix>;

public:
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  using acceleration_base_type::get_value;
  using displacement_base_type::get_value;
  using mass_inverse_base_type::get_value;
  using velocity_base_type::get_value;

  field_impl() = default;

  field_impl(
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_type,
      Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
          assembly_index_mapping);

  field_impl(const int nglob);

  template <specfem::sync::kind SyncField> void sync_fields() const;

  int nglob;

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_field() const {
    return displacement_base_type::template get_base_field_view<true>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_field_dot() const {
    return velocity_base_type::template get_base_field_view<true>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_field_dot_dot() const {
    return acceleration_base_type::template get_base_field_view<true>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_mass_inverse() const {
    return mass_inverse_base_type::template get_base_field_view<true>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_field() const {
    return displacement_base_type::template get_base_field_view<false>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_field_dot() const {
    return velocity_base_type::template get_base_field_view<false>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_field_dot_dot() const {
    return acceleration_base_type::template get_base_field_view<false>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::View<type_real **, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
  get_host_mass_inverse() const {
    return mass_inverse_base_type::template get_base_field_view<false>();
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
void deep_copy(const fields_impl::field_impl<DimensionTag, MediumTag> &dst,
               const fields_impl::field_impl<DimensionTag, MediumTag> &src) {

  Kokkos::deep_copy(dst.get_field(), src.get_field());
  Kokkos::deep_copy(dst.get_field_dot(), src.get_field_dot());
  Kokkos::deep_copy(dst.get_field_dot_dot(), src.get_field_dot_dot());
  Kokkos::deep_copy(dst.get_host_field(), src.get_host_field());
  Kokkos::deep_copy(dst.get_host_field_dot(), src.get_host_field_dot());
  Kokkos::deep_copy(dst.get_host_field_dot_dot(), src.get_host_field_dot_dot());

  return;
}

} // namespace fields_impl
} // namespace specfem::assembly
