#pragma once

#include "execution/for_each_level.hpp"
#include "specfem/point.hpp"

namespace specfem::assembly {

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load(const int iglob, const WavefieldType &field, ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;

  const auto &curr_field = field.template get_field<MediumTag>();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.displacement(icomp) =
          curr_field.template get_field<on_device>(iglob, icomp);
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.velocity(icomp) =
          curr_field.template get_field_dot<on_device>(iglob, icomp);
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.acceleration(icomp) =
          curr_field.template get_field_dot_dot<on_device>(iglob, icomp);
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < ViewType::components; ++icomp) {
      point_field.mass_matrix(icomp) =
          curr_field.template get_mass_inverse<on_device>(iglob, icomp);
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load(const typename ViewType::simd::mask_type &mask, const int *iglob,
          const WavefieldType &field, ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field = field.template get_field<MediumTag>();

  using simd_type = typename ViewType::simd::datatype;

  for (std::size_t icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      simd_type result([&](std::size_t lane) {
        return mask[lane] ? curr_field.template get_field<on_device>(
                                  iglob[lane], icomp)
                          : 0.0;
      });
      point_field.displacement(icomp) = result;
    }
    if constexpr (StoreVelocity) {
      simd_type result([&](std::size_t lane) {
        return mask[lane] ? curr_field.template get_field_dot<on_device>(
                                  iglob[lane], icomp)
                          : 0.0;
      });
      point_field.velocity(icomp) = result;
    }
    if constexpr (StoreAcceleration) {
      simd_type result([&](std::size_t lane) {
        return mask[lane] ? curr_field.template get_field_dot_dot<on_device>(
                                  iglob[lane], icomp)
                          : 0.0;
      });
      point_field.acceleration(icomp) = result;
    }
    if constexpr (StoreMassMatrix) {
      simd_type result([&](std::size_t lane) {
        return mask[lane] ? curr_field.template get_mass_inverse<on_device>(
                                  iglob[lane], icomp)
                          : 0.0;
      });
      point_field.mass_matrix(icomp) = result;
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load(const specfem::point::simd_assembly_index &index,
          const WavefieldType &field, ViewType &point_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const int iglob = index.iglob;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field = field.template get_field<MediumTag>();

  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.displacement(icomp))
          .copy_from(&(curr_field.template get_field<on_device>(iglob, icomp)),
                     tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.velocity(icomp))
          .copy_from(
              &(curr_field.template get_field_dot<on_device>(iglob, icomp)),
              tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_from(
              &(curr_field.template get_field_dot_dot<on_device>(iglob, icomp)),
              tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.mass_matrix(icomp))
          .copy_from(
              &(curr_field.template get_mass_inverse<on_device>(iglob, icomp)),
              tag_type());
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load(const specfem::point::index<ViewType::dimension_tag> &index,
          const WavefieldType &field, ViewType &point_field) {
  constexpr static auto MediumTag = ViewType::medium_tag;
  impl_load<on_device>(field.template get_iglob<on_device>(
                           index.ispec, index.iz, index.ix, MediumTag),
                       field, point_field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load(const specfem::point::assembly_index<false> &index,
          const WavefieldType &field, ViewType &point_field) {
  constexpr static auto MediumTag = ViewType::medium_tag;
  const int iglob = index.iglob;
  impl_load<on_device>(iglob, field, point_field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_load(const specfem::point::simd_index<ViewType::dimension_tag> &index,
          const WavefieldType &field, ViewType &point_field) {
  constexpr static auto MediumTag = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] = index.mask(lane)
                      ? field.template get_iglob<on_device>(
                            index.ispec + lane, index.iz, index.ix, MediumTag)
                      : field.nglob + 1;
  }

  impl_load<on_device>(mask, iglob, field, point_field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_store(const int iglob,
                                            const ViewType &point_field,
                                            const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;

  const auto &curr_field = field.template get_field<MediumTag>();

  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.template get_field<on_device>(iglob, icomp) =
          point_field.displacement(icomp);
    }
    if constexpr (StoreVelocity) {
      curr_field.template get_field_dot<on_device>(iglob, icomp) =
          point_field.velocity(icomp);
    }
    if constexpr (StoreAcceleration) {
      curr_field.template get_field_dot_dot<on_device>(iglob, icomp) =
          point_field.acceleration(icomp);
    }
    if constexpr (StoreMassMatrix) {
      curr_field.template get_mass_inverse<on_device>(iglob, icomp) =
          point_field.mass_matrix(icomp);
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store(const typename ViewType::simd::mask_type &mask, const int *iglob,
           const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field = field.template get_field<MediumTag>();
  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_field<on_device>(iglob_l, icomp) =
            point_field.displacement(icomp)[lane];
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_field_dot<on_device>(iglob_l, icomp) =
            point_field.velocity(icomp)[lane];
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_field_dot_dot<on_device>(iglob_l, icomp) =
            point_field.acceleration(icomp)[lane];
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_mass_inverse<on_device>(iglob_l, icomp) =
            point_field.mass_matrix(icomp)[lane];
      }
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store(const specfem::point::simd_assembly_index &index,
           const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field = field.template get_field<MediumTag>();
  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.displacement(icomp))
          .copy_to(&(curr_field.template get_field<on_device>(iglob, icomp)),
                   tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.velocity(icomp))
          .copy_to(
              &(curr_field.template get_field_dot<on_device>(iglob, icomp)),
              tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.acceleration(icomp))
          .copy_to(
              &(curr_field.template get_field_dot_dot<on_device>(iglob, icomp)),
              tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::Experimental::where(mask, point_field.mass_matrix(icomp))
          .copy_to(
              &(curr_field.template get_mass_inverse<on_device>(iglob, icomp)),
              tag_type());
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store(const specfem::point::index<ViewType::dimension_tag> &index,
           const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;

  impl_store<on_device>(field.template get_iglob<on_device>(
                            index.ispec, index.iz, index.ix, MediumTag),
                        point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store(const specfem::point::assembly_index<false> &index,
           const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;

  const int iglob = index.iglob;

  impl_store<on_device>(iglob, point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_store(const specfem::point::simd_index<ViewType::dimension_tag> &index,
           const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] = index.mask(lane)
                      ? field.template get_iglob<on_device>(
                            index.ispec + lane, index.iz, index.ix, MediumTag)
                      : field.nglob + 1;
  }

  impl_store<on_device>(mask, iglob, point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_add(const int iglob,
                                          const ViewType &point_field,
                                          const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;

  const auto &curr_field = field.template get_field<MediumTag>();
  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.template get_field<on_device>(iglob, icomp) +=
          point_field.displacement(icomp);
    }
    if constexpr (StoreVelocity) {
      curr_field.template get_field_dot<on_device>(iglob, icomp) +=
          point_field.velocity(icomp);
    }
    if constexpr (StoreAcceleration) {
      curr_field.template get_field_dot_dot<on_device>(iglob, icomp) +=
          point_field.acceleration(icomp);
    }
    if constexpr (StoreMassMatrix) {
      curr_field.template get_mass_inverse<on_device>(iglob, icomp) +=
          point_field.mass_matrix(icomp);
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add(const typename ViewType::simd::mask_type &mask, const int *iglob,
         const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field = field.template get_field<MediumTag>();
  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_field<on_device>(iglob_l, icomp) +=
            point_field.displacement(icomp)[lane];
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_field_dot<on_device>(iglob_l, icomp) +=
            point_field.velocity(icomp)[lane];
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_field_dot_dot<on_device>(iglob_l, icomp) +=
            point_field.acceleration(icomp)[lane];
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        curr_field.template get_mass_inverse<on_device>(iglob_l, icomp) +=
            point_field.mass_matrix(icomp)[lane];
      }
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add(const specfem::point::simd_assembly_index &index,
         const ViewType &point_field, const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  using mask_type = typename ViewType::simd::mask_type;
  using tag_type = typename ViewType::simd::tag_type;

  const int iglob = index.iglob;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &curr_field = field.template get_field<MediumTag>();
  if constexpr (StoreDisplacement) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &(curr_field.template get_field<on_device>(iglob, icomp)),
          tag_type());

      lhs += point_field.displacement(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &(curr_field.template get_field<on_device>(iglob, icomp)),
          tag_type());
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &(curr_field.template get_field_dot<on_device>(iglob, icomp)),
          tag_type());

      lhs += point_field.velocity(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &(curr_field.template get_field_dot<on_device>(iglob, icomp)),
          tag_type());
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &(curr_field.template get_field_dot_dot<on_device>(iglob, icomp)),
          tag_type());

      lhs += point_field.acceleration(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &(curr_field.template get_field_dot_dot<on_device>(iglob, icomp)),
          tag_type());
    }
  }

  if constexpr (StoreMassMatrix) {
    for (int icomp = 0; icomp < components; ++icomp) {
      typename ViewType::simd::datatype lhs;
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &(curr_field.template get_mass_inverse<on_device>(iglob, icomp)),
          tag_type());

      lhs += point_field.mass_matrix(icomp);
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &(curr_field.template get_mass_inverse<on_device>(iglob, icomp)),
          tag_type());
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add(const specfem::point::index<ViewType::dimension_tag> &index,
         const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;
  impl_add<on_device>(field.template get_iglob<on_device>(index.ispec, index.iz,
                                                          index.ix, MediumTag),
                      point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add(const specfem::point::assembly_index<false> &index,
         const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;

  const int iglob = index.iglob;

  impl_add<on_device>(iglob, point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_add(const specfem::point::simd_index<ViewType::dimension_tag> &index,
         const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] = index.mask(lane)
                      ? field.template get_iglob<on_device>(
                            index.ispec + lane, index.iz, index.ix, MediumTag)
                      : field.nglob + 1;
  }

  impl_add<on_device>(mask, iglob, point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_atomic_add(const int iglob,
                                                 const ViewType &point_field,
                                                 const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;

  const auto &curr_field = field.template get_field<MediumTag>();
  for (int icomp = 0; icomp < ViewType::components; ++icomp) {
    if constexpr (StoreDisplacement) {
      Kokkos::atomic_add(
          &(curr_field.template get_field<on_device>(iglob, icomp)),
          point_field.displacement(icomp));
    }
    if constexpr (StoreVelocity) {
      Kokkos::atomic_add(
          &(curr_field.template get_field_dot<on_device>(iglob, icomp)),
          point_field.velocity(icomp));
    }
    if constexpr (StoreAcceleration) {
      Kokkos::atomic_add(
          &(curr_field.template get_field_dot_dot<on_device>(iglob, icomp)),
          point_field.acceleration(icomp));
    }
    if constexpr (StoreMassMatrix) {
      Kokkos::atomic_add(
          &(curr_field.template get_mass_inverse<on_device>(iglob, icomp)),
          point_field.mass_matrix(icomp));
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_atomic_add(const typename ViewType::simd::mask_type &mask,
                const int *iglob, const ViewType &point_field,
                const WavefieldType &field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field = field.template get_field<MediumTag>();
  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    if (!mask[lane]) {
      continue;
    }

    const int iglob_l = iglob[lane];

    if constexpr (StoreDisplacement) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(
            &(curr_field.template get_field<on_device>(iglob_l, icomp)),
            point_field.displacement(icomp)[lane]);
      }
    }

    if constexpr (StoreVelocity) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(
            &(curr_field.template get_field_dot<on_device>(iglob_l, icomp)),
            point_field.velocity(icomp)[lane]);
      }
    }

    if constexpr (StoreAcceleration) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(
            &(curr_field.template get_field_dot_dot<on_device>(iglob_l, icomp)),
            point_field.acceleration(icomp)[lane]);
      }
    }

    if constexpr (StoreMassMatrix) {
      for (int icomp = 0; icomp < components; ++icomp) {
        Kokkos::atomic_add(
            &(curr_field.template get_mass_inverse<on_device>(iglob_l, icomp)),
            point_field.mass_matrix(icomp)[lane]);
      }
    }
  }

  return;
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  !ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_atomic_add(const specfem::point::index<ViewType::dimension_tag> &index,
                const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;

  impl_atomic_add<on_device>(field.template get_iglob<on_device>(
                                 index.ispec, index.iz, index.ix, MediumTag),
                             point_field, field);
}

template <bool on_device, typename WavefieldType, typename ViewType,
          typename std::enable_if_t<
              specfem::accessor::is_point_field<ViewType>::value &&
                  ViewType::simd::using_simd,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_atomic_add(
    const specfem::point::simd_index<ViewType::dimension_tag> &index,
    const ViewType &point_field, const WavefieldType &field) {

  constexpr static auto MediumTag = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  using mask_type = typename ViewType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] = index.mask(lane)
                      ? field.template get_iglob<on_device>(
                            index.ispec + lane, index.iz, index.ix, MediumTag)
                      : field.nglob + 1;
  }

  impl_atomic_add<on_device>(mask, iglob, point_field, field);
}

template <
    bool on_device, typename ChunkIndexType, typename WavefieldType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && !ViewType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_load(const ChunkIndexType &index,
                               const WavefieldType &field,
                               ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  const auto &curr_field = field.template get_field<MediumTag>();

  specfem::execution::for_each_level(
      index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto ielement = iterator_index.get_policy_index();
        const auto point_index = iterator_index.get_index();
        const int ispec = point_index.ispec;
        const int iz = point_index.iz;
        const int ix = point_index.ix;
        const int iglob =
            field.template get_iglob<on_device>(ispec, iz, ix, MediumTag);
        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            chunk_field.displacement(ielement, iz, ix, icomp) =
                curr_field.template get_field<on_device>(iglob, icomp);
          }
          if constexpr (StoreVelocity) {
            chunk_field.velocity(ielement, iz, ix, icomp) =
                curr_field.template get_field_dot<on_device>(iglob, icomp);
          }
          if constexpr (StoreAcceleration) {
            chunk_field.acceleration(ielement, iz, ix, icomp) =
                curr_field.template get_field_dot_dot<on_device>(iglob, icomp);
          }
          if constexpr (StoreMassMatrix) {
            chunk_field.mass_matrix(ielement, iz, ix, icomp) =
                curr_field.template get_mass_inverse<on_device>(iglob, icomp);
          }
        }
      });

  return;
}

template <
    bool on_device, typename ChunkIndexType, typename WavefieldType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && ViewType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_load(const ChunkIndexType &index,
                               const WavefieldType &field,
                               ViewType &chunk_field) {

  constexpr static bool StoreDisplacement = ViewType::store_displacement;
  constexpr static bool StoreVelocity = ViewType::store_velocity;
  constexpr static bool StoreAcceleration = ViewType::store_acceleration;
  constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
  constexpr static auto MediumTag = ViewType::medium_tag;
  constexpr static int components = ViewType::components;

  constexpr static int NGLL = ViewType::ngll;

  const auto &curr_field = field.template get_field<MediumTag>();

  using simd_type = typename ViewType::simd::datatype;

  specfem::execution::for_each_level(
      index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto ielement = iterator_index.get_policy_index();
        const auto point_index = iterator_index.get_index();
        const int ispec = point_index.ispec;
        const int iz = point_index.iz;
        const int ix = point_index.ix;

        for (std::size_t icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            simd_type result([&](std::size_t lane) {
              return point_index.mask(lane)
                         ? curr_field.template get_field<on_device>(
                               field.template get_iglob<on_device>(
                                   ispec + lane, iz, ix, MediumTag),
                               icomp)
                         : static_cast<type_real>(0.0);
            });
            chunk_field.displacement(ielement, iz, ix, icomp) = result;
          }
          if constexpr (StoreVelocity) {
            simd_type result([&](std::size_t lane) {
              return point_index.mask(lane)
                         ? curr_field.template get_field_dot<on_device>(
                               field.template get_iglob<on_device>(
                                   ispec + lane, iz, ix, MediumTag),
                               icomp)
                         : static_cast<type_real>(0.0);
            });
            chunk_field.velocity(ielement, iz, ix, icomp) = result;
          }
          if constexpr (StoreAcceleration) {
            simd_type result([&](std::size_t lane) {
              return point_index.mask(lane)
                         ? curr_field.template get_field_dot_dot<on_device>(
                               field.template get_iglob<on_device>(
                                   ispec + lane, iz, ix, MediumTag),
                               icomp)
                         : static_cast<type_real>(0.0);
            });
            chunk_field.acceleration(ielement, iz, ix, icomp) = result;
          }
          if constexpr (StoreMassMatrix) {
            simd_type result([&](std::size_t lane) {
              return point_index.mask(lane)
                         ? curr_field.template get_mass_inverse<on_device>(
                               field.template get_iglob<on_device>(
                                   ispec + lane, iz, ix, MediumTag),
                               icomp)
                         : static_cast<type_real>(0.0);
            });
            chunk_field.mass_matrix(ielement, iz, ix, icomp) = result;
          }
        }
      });

  return;
}

} // namespace specfem::assembly
