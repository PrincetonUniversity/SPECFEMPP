#pragma once

#include "point/assembly_index.hpp"
#include "point/coordinates.hpp"

namespace specfem {
namespace compute {

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load_on_device(const int iglob,
                                                     const WavefieldType &field,
                                                     ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.displacement(icomp) =
          curr_field.field(iglob, icomp);
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.velocity(icomp) =
          curr_field.field_dot(iglob, icomp);
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.acceleration(icomp) =
          curr_field.field_dot_dot(iglob, icomp);
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.mass_matrix(icomp) =
          curr_field.mass_inverse(iglob, icomp);
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load_on_device(const typename ViewType::simd::mask_type &mask,
                    const int *iglob, const WavefieldType &field,
                    ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.displacement(icomp)[lane] =
            curr_field.field(iglob_l, icomp);
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.velocity(icomp)[lane] =
            curr_field.field_dot(iglob_l, icomp);
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.acceleration(icomp)[lane] =
            curr_field.field_dot_dot(iglob_l, icomp);
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.mass_matrix(icomp)[lane] =
            curr_field.mass_inverse(iglob_l, icomp);
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load_on_device(const specfem::point::simd_assembly_index &index,
                    const WavefieldType &field, ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const int iglob = index.iglob;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.displacement(icomp))
          .copy_from(&curr_field.field(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.velocity(icomp))
          .copy_from(&curr_field.field_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_from(&curr_field.field_dot_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.mass_matrix(icomp))
          .copy_from(&curr_field.mass_inverse(iglob, icomp), tag_type());
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(const int iglob, const WavefieldType &field,
                       ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.displacement(icomp) =
          curr_field.h_field(iglob, icomp);
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.velocity(icomp) =
          curr_field.h_field_dot(iglob, icomp);
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.acceleration(icomp) =
          curr_field.h_field_dot_dot(iglob, icomp);
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.mass_matrix(icomp) =
          curr_field.h_mass_inverse(iglob, icomp);
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    std::enable_if_t<ViewType::isPointFieldType && ViewType::simd::using_simd,
                     int> = 0>
inline void impl_load_on_host(const typename ViewType::simd::mask_type &mask,
                       const int *iglob, const WavefieldType &field,
                       ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.displacement(icomp)[lane] =
            curr_field.h_field(iglob_l, icomp);
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.velocity(icomp)[lane] =
            curr_field.h_field_dot(iglob_l, icomp);
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.acceleration(icomp)[lane] =
            curr_field.h_field_dot_dot(iglob_l, icomp);
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        point_field.mass_matrix(icomp)[lane] =
            curr_field.h_mass_inverse(iglob_l, icomp);
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    std::enable_if_t<ViewType::isPointFieldType && ViewType::simd::using_simd,
                     int> = 0>
inline void impl_load_on_host(const specfem::point::simd_assembly_index &index,
                       const WavefieldType &field, ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.displacement(icomp))
          .copy_from(&curr_field.h_field(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.velocity(icomp))
          .copy_from(&curr_field.h_field_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_from(&curr_field.h_field_dot_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.mass_matrix(icomp))
          .copy_from(&curr_field.h_mass_inverse(iglob, icomp), tag_type());
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load_on_device(const specfem::point::index<ViewType::dimension> &index,
                    const WavefieldType &field, ViewType &point_field) {
  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob_l = field.index_mapping(index.ispec, index.iz, index.ix);
  const int iglob =
      field.assembly_index_mapping(iglob_l, static_cast<int>(MediumType));
  impl_load_on_device(iglob, field, point_field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load_on_device(const specfem::point::assembly_index<false> &index,
                    const WavefieldType &field, ViewType &point_field) {
  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob = index.iglob;
  impl_load_on_device(iglob, field, point_field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load_on_device(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const WavefieldType &field, ViewType &point_field) {
  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.assembly_index_mapping(
                  field.index_mapping(index.ispec + lane, index.iz, index.ix),
                  static_cast<int>(MediumType))
            : field.nglob + 1;
  }

  impl_load_on_device(mask, iglob, field, point_field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(const specfem::point::index<ViewType::dimension> &index,
                       const WavefieldType &field, ViewType &point_field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_load_on_host(iglob, field, point_field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(const specfem::point::assembly_index<false> &index,
                       const WavefieldType &field, ViewType &point_field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob = index.iglob;
  impl_load_on_host(iglob, field, point_field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const WavefieldType &field, ViewType &point_field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_assembly_index_mapping(
                  field.h_index_mapping(index.ispec + lane, index.iz, index.ix),
                  static_cast<int>(MediumType))
            : field.nglob + 1;
  }

  impl_load_on_host(mask, iglob, field, point_field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store_on_device(const int iglob, const ViewType &point_field,
                     const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.field(iglob, icomp) = point_field.displacement(icomp);
    }
    if constexpr (StoreVelocity) {
      curr_field.field_dot(iglob, icomp) = point_field.velocity(icomp);
    }
    if constexpr (StoreAcceleration) {
      curr_field.field_dot_dot(iglob, icomp) = point_field.acceleration(icomp);
    }
    if constexpr (StoreMassMatrix) {
      curr_field.mass_inverse(iglob, icomp) = point_field.mass_matrix(icomp);
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store_on_device(const typename ViewType::simd::mask_type &mask,
                     const int *iglob, const ViewType &point_field,
                     const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.field(iglob_l, icomp) =
            point_field.displacement(icomp)[lane];
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.field_dot(iglob_l, icomp) =
            point_field.velocity(icomp)[lane];
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.field_dot_dot(iglob_l, icomp) =
            point_field.acceleration(icomp)[lane];
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.mass_inverse(iglob_l, icomp) =
            point_field.mass_matrix(icomp)[lane];
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store_on_device(const specfem::point::simd_assembly_index &index,
                     const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.displacement(icomp))
          .copy_to(&curr_field.h_field(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.velocity(icomp))
          .copy_to(&curr_field.field_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_to(&curr_field.field_dot_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.mass_matrix(icomp))
          .copy_to(&curr_field.mass_inverse(iglob, icomp), tag_type());
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_store_on_host(const int iglob, const ViewType &point_field,
                        const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.h_field(iglob, icomp) = point_field.displacement(icomp);
    }
    if constexpr (StoreVelocity) {
      curr_field.h_field_dot(iglob, icomp) = point_field.velocity(icomp);
    }
    if constexpr (StoreAcceleration) {
      curr_field.h_field_dot_dot(iglob, icomp) =
          point_field.acceleration(icomp);
    }
    if constexpr (StoreMassMatrix) {
      curr_field.h_mass_inverse(iglob, icomp) = point_field.mass_matrix(icomp);
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_store_on_host(const typename ViewType::simd::mask_type &mask,
                        const int *iglob, const ViewType &point_field,
                        const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_field(iglob_l, icomp) =
            point_field.displacement(icomp)[lane];
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_field_dot(iglob_l, icomp) =
            point_field.velocity(icomp)[lane];
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_field_dot_dot(iglob_l, icomp) =
            point_field.acceleration(icomp)[lane];
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_mass_inverse(iglob_l, icomp) =
            point_field.mass_matrix(icomp)[lane];
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_store_on_host(const specfem::point::simd_assembly_index &index,
                        const ViewType &point_field,
                        const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.displacement(icomp))
          .copy_to(&curr_field.h_field(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.velocity(icomp))
          .copy_to(&curr_field.h_field_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_to(&curr_field.h_field_dot_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_to(&curr_field.h_mass_matrix(iglob, icomp), tag_type());
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store_on_device(const specfem::point::index<ViewType::dimension> &index,
                     const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_store_on_device(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store_on_device(const specfem::point::assembly_index<false> &index,
                     const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  impl_store_on_device(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_store_on_device(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.assembly_index_mapping(
                  field.index_mapping(index.ispec + lane, index.iz, index.ix),
                  static_cast<int>(MediumType))
            : field.nglob + 1;
  }

  impl_store_on_device(mask, iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_store_on_host(const specfem::point::index<ViewType::dimension> &index,
                        const ViewType &point_field,
                        const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_store_on_host(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_store_on_host(const specfem::point::assembly_index<false> &index,
                        const ViewType &point_field,
                        const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  impl_store_on_host(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_store_on_host(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  impl_store_on_host(mask, iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add_on_device(const int iglob, const ViewType &point_field,
                   const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.field(iglob, icomp) += point_field.displacement(icomp);
    }
    if constexpr (StoreVelocity) {
      curr_field.field_dot(iglob, icomp) += point_field.velocity(icomp);
    }
    if constexpr (StoreAcceleration) {
      curr_field.field_dot_dot(iglob, icomp) += point_field.acceleration(icomp);
    }
    if constexpr (StoreMassMatrix) {
      curr_field.mass_inverse(iglob, icomp) += point_field.mass_matrix(icomp);
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add_on_device(const typename ViewType::simd::mask_type &mask,
                   const int *iglob, const ViewType &point_field,
                   const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.field(iglob_l, icomp) +=
            point_field.displacement(icomp)[lane];
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.field_dot(iglob_l, icomp) +=
            point_field.velocity(icomp)[lane];
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.field_dot_dot(iglob_l, icomp) +=
            point_field.acceleration(icomp)[lane];
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.mass_inverse(iglob_l, icomp) +=
            point_field.mass_matrix(icomp)[lane];
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add_on_device(const specfem::point::simd_assembly_index &index,
                   const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.field(iglob, icomp), tag_type());

      lhs += point_field.displacement(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.field(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.field_dot(iglob, icomp), tag_type());

      lhs += point_field.velocity(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.field_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.field_dot_dot(iglob, icomp), tag_type());

      lhs += point_field.acceleration(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.field_dot_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.mass_inverse(iglob, icomp), tag_type());

      lhs += point_field.mass_matrix(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.mass_inverse(iglob, icomp), tag_type());
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_add_on_host(const int iglob, const ViewType &point_field,
                      const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.h_field(iglob, icomp) += point_field.displacement(icomp);
    }
    if constexpr (StoreVelocity) {
      curr_field.h_field_dot(iglob, icomp) += point_field.velocity(icomp);
    }
    if constexpr (StoreAcceleration) {
      curr_field.h_field_dot_dot(iglob, icomp) +=
          point_field.acceleration(icomp);
    }
    if constexpr (StoreMassMatrix) {
      curr_field.h_mass_inverse(iglob, icomp) += point_field.mass_matrix(icomp);
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_add_on_host(const typename ViewType::simd::mask_type &mask,
                      const int *iglob, const ViewType &point_field,
                      const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_field(iglob_l, icomp) +=
            point_field.displacement(icomp)[lane];
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_field_dot(iglob_l, icomp) +=
            point_field.velocity(icomp)[lane];
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_field_dot_dot(iglob_l, icomp) +=
            point_field.acceleration(icomp)[lane];
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.h_mass_inverse(iglob_l, icomp) +=
            point_field.mass_matrix(icomp)[lane];
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_add_on_host(const specfem::point::simd_assembly_index &index,
                      const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.h_field(iglob, icomp), tag_type());

      lhs += point_field.displacement(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.h_field(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.h_field_dot(iglob, icomp), tag_type());

      lhs += point_field.velocity(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.h_field_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.h_field_dot_dot(iglob, icomp), tag_type());

      lhs += point_field.acceleration(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.h_field_dot_dot(iglob, icomp), tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &curr_field.h_mass_inverse(iglob, icomp), tag_type());

      lhs += point_field.mass_matrix(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &curr_field.h_mass_inverse(iglob, icomp), tag_type());
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add_on_device(const specfem::point::index<ViewType::dimension> &index,
                   const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_add_on_device(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add_on_device(const specfem::point::assembly_index<false> &index,
                   const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  impl_add_on_device(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add_on_device(const specfem::point::simd_index<ViewType::dimension> &index,
                   const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.assembly_index_mapping(
                  field.index_mapping(index.ispec + lane, index.iz, index.ix),
                  static_cast<int>(MediumType))
            : field.nglob + 1;
  }

  impl_add_on_device(mask, iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_add_on_host(const specfem::point::index<ViewType::dimension> &index,
                      const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_add_on_host(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_add_on_host(const specfem::point::assembly_index<false> &index,
                      const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  impl_add_on_host(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_add_on_host(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  impl_add_on_host(mask, iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_atomic_add_on_device(const int iglob, const ViewType &point_field,
                          const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      Kokkos::atomic_add(&curr_field.field(iglob, icomp),
                         point_field.displacement(icomp));
    }
    if constexpr (StoreVelocity) {
      Kokkos::atomic_add(&curr_field.field_dot(iglob, icomp),
                         point_field.velocity(icomp));
    }
    if constexpr (StoreAcceleration) {
      Kokkos::atomic_add(&curr_field.field_dot_dot(iglob, icomp),
                         point_field.acceleration(icomp));
    }
    if constexpr (StoreMassMatrix) {
      Kokkos::atomic_add(&curr_field.mass_inverse(iglob, icomp),
                         point_field.mass_matrix(icomp));
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_atomic_add_on_device(const typename ViewType::simd::mask_type &mask,
                          const int *iglob, const ViewType &point_field,
                          const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.field(iglob_l, icomp),
                           point_field.displacement(icomp)[lane]);
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.field_dot(iglob_l, icomp),
                           point_field.velocity(icomp)[lane]);
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.field_dot_dot(iglob_l, icomp),
                           point_field.acceleration(icomp)[lane]);
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.mass_inverse(iglob_l, icomp),
                           point_field.mass_matrix(icomp)[lane]);
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_atomic_add_on_host(const int iglob, const ViewType &point_field,
                             const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      Kokkos::atomic_add(&curr_field.h_field(iglob, icomp),
                         point_field.displacement(icomp));
    }
    if constexpr (StoreVelocity) {
      Kokkos::atomic_add(&curr_field.h_field_dot(iglob, icomp),
                         point_field.velocity(icomp));
    }
    if constexpr (StoreAcceleration) {
      Kokkos::atomic_add(&curr_field.h_field_dot_dot(iglob, icomp),
                         point_field.acceleration(icomp));
    }
    if constexpr (StoreMassMatrix) {
      Kokkos::atomic_add(&curr_field.h_mass_inverse(iglob, icomp),
                         point_field.mass_matrix(icomp));
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_atomic_add_on_host(const typename ViewType::simd::mask_type &mask,
                             const int *iglob, const ViewType &point_field,
                             const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.h_field(iglob_l, icomp),
                           point_field.displacement(icomp)[lane]);
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.h_field_dot(iglob_l, icomp),
                           point_field.velocity(icomp)[lane]);
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.h_field_dot_dot(iglob_l, icomp),
                           point_field.acceleration(icomp)[lane]);
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(&curr_field.h_mass_inverse(iglob_l, icomp),
                           point_field.mass_matrix(icomp)[lane]);
      }
    }
  }

  return;
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void impl_atomic_add_on_device(
    const specfem::point::index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_atomic_add_on_device(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void impl_atomic_add_on_device(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.assembly_index_mapping(
                  field.index_mapping(index.ispec + lane, index.iz, index.ix),
                  static_cast<int>(MediumType))
            : field.nglob + 1;
  }

  impl_atomic_add_on_device(mask, iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_atomic_add_on_host(
    const specfem::point::index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  impl_atomic_add_on_host(iglob, point_field, field);
}

template <
    typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_atomic_add_on_host(
    const specfem::point::simd_index<ViewType::dimension> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  impl_atomic_add_on_host(mask, iglob, point_field, field);
}

template <
    typename MemberType, typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isElementFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load_on_device(const MemberType &team, const int ispec,
                    const WavefieldType &field, ViewType &element_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a device execution space");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int iz, ix;
        sub2ind(xz, NGLL, iz, ix);
        const int iglob = field.assembly_index_mapping(
            field.index_mapping(ispec, iz, ix), static_cast<int>(MediumType));

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            element_field.displacement(iz, ix, icomp) =
                curr_field.field(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            element_field.velocity(iz, ix, icomp) =
                curr_field.field_dot(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            element_field.acceleration(iz, ix, icomp) =
                curr_field.field_dot_dot(iglob, icomp);
          }
          if constexpr (StoreMassMatrix) {
            element_field.mass_matrix(iz, ix, icomp) =
                curr_field.mass_inverse(iglob, icomp);
          }
        }
      });

  return;
}

template <
    typename MemberType, typename WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isElementFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(const MemberType &team, const int ispec,
                       const WavefieldType &field, ViewType &element_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  static_assert(
      std::is_same_v<typename MemberType::execution_space, Kokkos::HostSpace>,
      "Calling team must have a host execution space");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int iz, ix;
        sub2ind(xz, NGLL, iz, ix);
        const int iglob = field.h_assembly_index_mapping(
            field.h_index_mapping(ispec, iz, ix), static_cast<int>(MediumType));

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            element_field.displacement(iz, ix, icomp) =
                curr_field.h_field(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            element_field.velocity(iz, ix, icomp) =
                curr_field.h_field_dot(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            element_field.acceleration(iz, ix, icomp) =
                curr_field.h_field_dot_dot(iglob, icomp);
          }
          if constexpr (StoreMassMatrix) {
            element_field.mass_matrix(iz, ix, icomp) =
                curr_field.h_mass_inverse(iglob, icomp);
          }
        }
      });

  return;
}

template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
impl_load_on_device(const MemberType &team, const IteratorType &iterator,
                    const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;
  constexpr static bool using_simd = ViewType::simd::using_simd;

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a device execution space");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const int ielement = iterator_index.ielement;
        const int ispec = iterator_index.index.ispec;
        const int iz = iterator_index.index.iz;
        const int ix = iterator_index.index.ix;

        const int iglob = field.assembly_index_mapping(
            field.index_mapping(ispec, iz, ix), static_cast<int>(MediumType));

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            chunk_field.displacement(ielement, iz, ix, icomp) =
                curr_field.field(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            chunk_field.velocity(ielement, iz, ix, icomp) =
                curr_field.field_dot(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            chunk_field.acceleration(ielement, iz, ix, icomp) =
                curr_field.field_dot_dot(iglob, icomp);
          }
          if constexpr (StoreMassMatrix) {
            chunk_field.mass_matrix(ielement, iz, ix, icomp) =
                curr_field.mass_inverse(iglob, icomp);
          }
        }
      });

  return;
}

template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
impl_load_on_device(const MemberType &team, const IteratorType &iterator,
                    const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a device execution space");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const int ielement = iterator_index.ielement;
        const int ispec = iterator_index.index.ispec;
        const int iz = iterator_index.index.iz;
        const int ix = iterator_index.index.ix;

        for (int lane = 0; lane < IteratorType::simd::size(); ++lane) {
          if (!iterator_index.index.mask(lane)) {
            continue;
          }

          const int iglob = field.assembly_index_mapping(
              field.index_mapping(ispec + lane, iz, ix),
              static_cast<int>(MediumType));

          for (int icomp = 0; icomp < components; ++icomp) {
            if constexpr (StoreDisplacement) {
              chunk_field.displacement(ielement, iz, ix, icomp)[lane] =
                  curr_field.field(iglob, icomp);
            }
            if constexpr (StoreVelocity) {
              chunk_field.velocity(ielement, iz, ix, icomp)[lane] =
                  curr_field.field_dot(iglob, icomp);
            }
            if constexpr (StoreAcceleration) {
              chunk_field.acceleration(ielement, iz, ix, icomp)[lane] =
                  curr_field.field_dot_dot(iglob, icomp);
            }
            if constexpr (StoreMassMatrix) {
              chunk_field.mass_matrix(ielement, iz, ix, icomp)[lane] =
                  curr_field.mass_inverse(iglob, icomp);
            }
          }
        }
      });

  return;
}

template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && !ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(const MemberType &team, const IteratorType &iterator,
                       const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  static_assert(
      std::is_same_v<typename MemberType::execution_space, Kokkos::HostSpace>,
      "Calling team must have a host execution space");

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const int ielement = iterator_index.ielement;
        const int ispec = iterator_index.index.ispec;
        const int iz = iterator_index.index.iz;
        const int ix = iterator_index.index.ix;

        const int iglob = field.h_assembly_index_mapping(
            field.h_index_mapping(ispec, iz, ix), static_cast<int>(MediumType));

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            chunk_field.displacement(ielement, iz, ix, icomp) =
                curr_field.h_field(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            chunk_field.velocity(ielement, iz, ix, icomp) =
                curr_field.h_field_dot(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            chunk_field.acceleration(ielement, iz, ix, icomp);
          }
          if constexpr (StoreMassMatrix) {
            chunk_field.mass_matrix(ielement, iz, ix, icomp) =
                curr_field.h_mass_inverse(iglob, icomp);
          }
        }
      });

  return;
}

template <
    typename MemberType, typename WavefieldType, typename IteratorType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && ViewType::simd::using_simd, int> = 0>
inline void impl_load_on_host(const MemberType &team, const IteratorType &iterator,
                       const WavefieldType &field, ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumType = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  static_assert(
      std::is_same_v<typename MemberType::execution_space, Kokkos::HostSpace>,
      "Calling team must have a host execution space");

  static_assert(
      std::is_same_v<typename IteratorType::simd, typename ViewType::simd>,
      "Iterator and View must have the same simd type");

  static_assert(
      Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                 typename ViewType::memory_space>::accessible,
      "Calling team must have access to the memory space of the view");

  const auto &curr_field =
      [&]() -> const specfem::compute::impl::field_impl<
                specfem::dimension::type::dim2, MediumType> & {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const int ielement = iterator_index.ielement;
        const int ispec = iterator_index.index.ispec;
        const int iz = iterator_index.index.iz;
        const int ix = iterator_index.index.ix;

        for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
          if (!iterator_index.index.mask(lane)) {
            continue;
          }

          const int iglob = field.h_assembly_index_mapping(
              field.h_index_mapping(ispec + lane, iz, ix),
              static_cast<int>(MediumType));

          for (int icomp = 0; icomp < components; ++icomp) {
            if constexpr (StoreDisplacement) {
              chunk_field.displacement(ielement, iz, ix, icomp)[lane] =
                  curr_field.h_field(iglob, icomp);
            }
            if constexpr (StoreVelocity) {
              chunk_field.velocity(ielement, iz, ix, icomp)[lane] =
                  curr_field.h_field_dot(iglob, icomp);
            }
            if constexpr (StoreAcceleration) {
              chunk_field.acceleration(ielement, iz, ix, icomp)[lane] =
                  curr_field.h_field_dot_dot(iglob, icomp);
            }
            if constexpr (StoreMassMatrix) {
              chunk_field.mass_matrix(ielement, iz, ix, icomp)[lane] =
                  curr_field.h_mass_inverse(iglob, icomp);
            }
          }
        }
      });

  return;
}

} // namespace compute
} // namespace specfem
