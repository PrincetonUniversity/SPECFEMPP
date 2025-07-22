#pragma once

#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

namespace fields_impl {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class base_field {
private:
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  int nglob;

protected:
  base_field() = default;
  base_field(const int nglob, std::string name)
      : nglob(nglob), data(name, nglob, components),
        h_data(Kokkos::create_mirror_view(data)) {}

  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &
  get_base_field(const int &iglob, const int &icomp) const {
    if constexpr (on_device) {
      return data(iglob, icomp);
    } else {
      return h_data(iglob, icomp);
    }
  }

  template <specfem::sync::kind SyncField> void sync() const {
    if constexpr (SyncField == specfem::sync::kind::HostToDevice) {
      Kokkos::deep_copy(data, h_data);
    } else if constexpr (SyncField == specfem::sync::kind::DeviceToHost) {
      Kokkos::deep_copy(h_data, data);
    }
  }

private:
  using ViewType = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;
  ViewType data;
  ViewType::HostMirror h_data;

protected:
  ViewType get_data() const { return data; }
  ViewType::HostMirror get_host_data() const { return h_data; }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class field : public base_field<DimensionTag, MediumTag> {
public:
  field() = default;
  field(int nglob)
      : base_field<DimensionTag, MediumTag>(
            nglob, "specfem::assembly::fields::field") {}

  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &
  get_field(const int &iglob, const int &icomp) const {
    return this->template get_base_field<on_device>(iglob, icomp);
  }

  auto get_field() const { return this->get_data(); }
  auto get_host_field() const { return this->get_host_data(); }

  template <specfem::sync::kind SyncField> void sync() const {
    base_field<DimensionTag, MediumTag>::template sync<SyncField>();
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class field_dot : public base_field<DimensionTag, MediumTag> {
public:
  field_dot() = default;
  field_dot(int nglob)
      : base_field<DimensionTag, MediumTag>(
            nglob, "specfem::assembly::fields::field_dot") {}

  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &
  get_field_dot(const int &iglob, const int &icomp) const {
    return this->template get_base_field<on_device>(iglob, icomp);
  }

  auto get_field_dot() const { return this->get_data(); }
  auto get_host_field_dot() const { return this->get_host_data(); }

  template <specfem::sync::kind SyncField> void sync() const {
    base_field<DimensionTag, MediumTag>::template sync<SyncField>();
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class field_dot_dot : public base_field<DimensionTag, MediumTag> {
public:
  field_dot_dot() = default;
  field_dot_dot(int nglob)
      : base_field<DimensionTag, MediumTag>(
            nglob, "specfem::assembly::fields::field_dot_dot") {}

  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &
  get_field_dot_dot(const int &iglob, const int &icomp) const {
    return this->template get_base_field<on_device>(iglob, icomp);
  }

  auto get_field_dot_dot() const { return this->get_data(); }
  auto get_host_field_dot_dot() const { return this->get_host_data(); }

  template <specfem::sync::kind SyncField> void sync() const {
    base_field<DimensionTag, MediumTag>::template sync<SyncField>();
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class mass_inverse : public base_field<DimensionTag, MediumTag> {
public:
  mass_inverse() = default;
  mass_inverse(int nglob)
      : base_field<DimensionTag, MediumTag>(
            nglob, "specfem::assembly::fields::mass_inverse") {}

  auto get_mass_inverse() const { return this->get_data(); }
  auto get_host_mass_inverse() const { return this->get_host_data(); }

  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION constexpr type_real &
  get_mass_inverse(const int &iglob, const int &icomp) const {
    return this->template get_base_field<on_device>(iglob, icomp);
  }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class field_impl : public field<DimensionTag, MediumTag>,
                   public field_dot<DimensionTag, MediumTag>,
                   public field_dot_dot<DimensionTag, MediumTag>,
                   public mass_inverse<DimensionTag, MediumTag> {
public:
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  field_impl() = default;

  field_impl(
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_type,
      Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
          assembly_index_mapping);

  field_impl(const int nglob);

  template <specfem::sync::kind SyncField> void sync_fields() const;

  int nglob;
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
