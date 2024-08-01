#ifndef _COMPUTE_FIELDS_SIMULATION_FIELD_HPP_
#define _COMPUTE_FIELDS_SIMULATION_FIELD_HPP_

#include "compute/fields/impl/field_impl.hpp"
#include "element/field.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
#include "point/assembly_index.hpp"
#include "point/field.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {

template <specfem::wavefield::type WavefieldType> struct simulation_field {

  using elastic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;

  using acoustic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;

  simulation_field() = default;

  simulation_field(const specfem::compute::mesh &mesh,
                   const specfem::compute::properties &properties);

  // template <specfem::element::medium_tag medium>
  // KOKKOS_INLINE_FUNCTION
  //     specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //     medium> get_field() const {
  //   if constexpr (medium == specfem::element::medium_tag::elastic) {
  //     return elastic;
  //   } else if constexpr (medium == specfem::element::medium_tag::acoustic) {
  //     return acoustic;
  //   } else {
  //     static_assert("medium type not supported");
  //   }
  // }

  void copy_to_host() { sync_fields<specfem::sync::kind::DeviceToHost>(); }

  void copy_to_device() { sync_fields<specfem::sync::kind::HostToDevice>(); }

  template <specfem::wavefield::type DestinationWavefieldType>
  void operator=(const simulation_field<DestinationWavefieldType> &rhs) {
    this->nglob = rhs.nglob;
    this->assembly_index_mapping = rhs.assembly_index_mapping;
    this->h_assembly_index_mapping = rhs.h_assembly_index_mapping;
    this->elastic = rhs.elastic;
    this->acoustic = rhs.acoustic;
  }

  template <specfem::element::medium_tag MediumType>
  KOKKOS_INLINE_FUNCTION int get_nglob() const {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return elastic.nglob;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return acoustic.nglob;
    } else {
      static_assert("medium type not supported");
    }
  }

  using ViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  int nglob = 0;
  int nspec;
  int ngllz;
  int ngllx;
  ViewType index_mapping;
  ViewType::HostMirror h_index_mapping;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::DevMemSpace>
      assembly_index_mapping;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      h_assembly_index_mapping;
  specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic>
      elastic;
  specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::acoustic>
      acoustic;

private:
  template <specfem::sync::kind sync> void sync_fields() {
    elastic.sync_fields<sync>();
    acoustic.sync_fields<sync>();
  }
};

template <specfem::wavefield::type WavefieldType1,
          specfem::wavefield::type WavefieldType2>
void deep_copy(simulation_field<WavefieldType1> &dst,
               const simulation_field<WavefieldType2> &src) {
  dst.nglob = src.nglob;
  Kokkos::deep_copy(dst.assembly_index_mapping, src.assembly_index_mapping);
  Kokkos::deep_copy(dst.h_assembly_index_mapping, src.h_assembly_index_mapping);
  specfem::compute::deep_copy(dst.elastic, src.elastic);
  specfem::compute::deep_copy(dst.acoustic, src.acoustic);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const int iglob,
               const specfem::compute::simulation_field<WavefieldType> &field,
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
    point_field.displacement =
        Kokkos::subview(curr_field.field, iglob, Kokkos::ALL);
  }

  if constexpr (StoreVelocity) {
    point_field.velocity =
        Kokkos::subview(curr_field.field_dot, iglob, Kokkos::ALL);
  }

  if constexpr (StoreAcceleration) {
    point_field.acceleration =
        Kokkos::subview(curr_field.field_dot_dot, iglob, Kokkos::ALL);
  }

  if constexpr (StoreMassMatrix) {
    point_field.mass_matrix =
        Kokkos::subview(curr_field.mass_inverse, iglob, Kokkos::ALL);
  }

  return;
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const int *iglob,
               const specfem::compute::simulation_field<WavefieldType> &field,
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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      return;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const specfem::point::simd_assembly_index &index,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &point_field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const int iglob,
    const specfem::compute::simulation_field<WavefieldType> &field,
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
    point_field.displacement =
        Kokkos::subview(curr_field.h_field, iglob, Kokkos::ALL);
  }

  if constexpr (StoreVelocity) {
    point_field.velocity =
        Kokkos::subview(curr_field.h_field_dot, iglob, Kokkos::ALL);
  }

  if constexpr (StoreAcceleration) {
    point_field.acceleration =
        Kokkos::subview(curr_field.h_field_dot_dot, iglob, Kokkos::ALL);
  }

  if constexpr (StoreMassMatrix) {
    point_field.mass_matrix =
        Kokkos::subview(curr_field.h_mass_inverse, iglob, Kokkos::ALL);
  }

  return;
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    std::enable_if_t<ViewType::isPointFieldType && ViewType::simd::using_simd,
                     int> = 0>
void load_on_host(
    const int *iglob,
    const specfem::compute::simulation_field<WavefieldType> &field,
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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    std::enable_if_t<ViewType::isPointFieldType && ViewType::simd::using_simd,
                     int> = 0>
void load_on_host(
    const specfem::point::simd_assembly_index &index,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &point_field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const specfem::point::index &index,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &point_field) {
  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob_l = field.index_mapping(index.ispec, index.iz, index.ix);
  const int iglob =
      field.assembly_index_mapping(iglob_l, static_cast<int>(MediumType));
  load_on_device(iglob, field, point_field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const specfem::point::assembly_index &index,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &point_field) {
  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob = index.iglob;
  load_on_device(iglob, field, point_field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const specfem::point::simd_index &index,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &point_field) {
  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  load_on_device(iglob, field, point_field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const specfem::point::index &index,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &point_field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  load_on_host(iglob, field, point_field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const specfem::point::assembly_index &index,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &point_field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  const int iglob = index.iglob;
  load_on_host(iglob, field, point_field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const specfem::point::simd_index &index,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &point_field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  load_on_host(iglob, field, point_field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void store_on_device(
    const int iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void store_on_device(
    const int *iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void store_on_device(
    const specfem::point::simd_assembly_index &index,
    const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void store_on_host(
    const int iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void store_on_host(
    const int *iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void store_on_host(
    const specfem::point::simd_assembly_index &index,
    const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void store_on_device(
    const specfem::point::index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  store_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void store_on_device(
    const specfem::point::assembly_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  store_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void store_on_device(
    const specfem::point::simd_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  store_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void store_on_host(
    const specfem::point::index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  store_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void store_on_host(
    const specfem::point::assembly_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  store_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void store_on_host(
    const specfem::point::simd_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  store_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
add_on_device(const int iglob, const ViewType &point_field,
              const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
add_on_device(const int *iglob, const ViewType &point_field,
              const specfem::compute::simulation_field<WavefieldType> &field) {

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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
add_on_device(const specfem::point::simd_assembly_index &index,
              const ViewType &point_field,
              const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void add_on_host(
    const int iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void add_on_host(
    const int *iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void add_on_host(
    const specfem::point::simd_assembly_index &index,
    const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
add_on_device(const specfem::point::index &index, const ViewType &point_field,
              const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  add_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
add_on_device(const specfem::point::assembly_index &index,
              const ViewType &point_field,
              const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  add_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
add_on_device(const specfem::point::simd_index &index,
              const ViewType &point_field,
              const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  add_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void add_on_host(
    const specfem::point::index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  add_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void add_on_host(
    const specfem::point::assembly_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = index.iglob;

  add_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void add_on_host(
    const specfem::point::simd_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  add_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void atomic_add_on_device(
    const int iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void atomic_add_on_device(
    const int *iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void atomic_add_on_host(
    const int iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void atomic_add_on_host(
    const int *iglob, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

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
    const int iglob_l = iglob[lane];
    if (iglob_l >= field.nglob) {
      continue;
    }

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
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void atomic_add_on_device(
    const specfem::point::index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  atomic_add_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void atomic_add_on_device(
    const specfem::point::simd_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  atomic_add_on_device(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && !ViewType::simd::using_simd, int> = 0>
void atomic_add_on_host(
    const specfem::point::index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;

  const int iglob = field.h_assembly_index_mapping(
      field.h_index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  atomic_add_on_host(iglob, point_field, field);
}

template <
    specfem::wavefield::type WavefieldType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isPointFieldType && ViewType::simd::using_simd, int> = 0>
void atomic_add_on_host(
    const specfem::point::simd_index &index, const ViewType &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr static auto MediumType = ViewType::medium_tag;
  int iglob[ViewType::simd::size()];

  for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
    iglob[lane] =
        (index.mask(std::size_t(lane)))
            ? field.h_index_mapping(index.ispec + lane, index.iz, index.ix)
            : field.nglob + 1;
  }

  atomic_add_on_host(iglob, point_field, field);
}

template <
    typename MemberType, specfem::wavefield::type WavefieldType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isElementFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const MemberType &team, const int ispec,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &element_field) {

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

// template <
//     typename MemberType, specfem::wavefield::type WavefieldType,
//     typename ViewType,
//     typename std::enable_if_t<ViewType::isElementFieldType &&
//     ViewType::simd::using_simd,
//                      int> = 0>
// NOINLINE KOKKOS_FUNCTION void
// load_on_device(const MemberType &team, const int ispec,
//                const specfem::compute::simulation_field<WavefieldType>
//                &field, ViewType &element_field) {

//   constexpr static bool StoreDisplacement = ViewType::store_displacement;
//   constexpr static bool StoreVelocity = ViewType::store_velocity;
//   constexpr static bool StoreAcceleration = ViewType::store_acceleration;
//   constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
//   constexpr static auto MediumType = ViewType::medium_tag;
//   constexpr static int components = ViewType::components;

//   constexpr static int NGLL = ViewType::ngll;

//   static_assert(std::is_same_v<typename MemberType::execution_space,
//                                Kokkos::DefaultExecutionSpace>,
//                 "Calling team must have a device execution space");

//   constexpr auto& curr_field = [&]() constexp
//       -> specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
//                                             MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
//         for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
//           if (ispec + lane >= field.nspec) {
//             continue;
//           }

//           int iz, ix;
//           sub2ind(xz, NGLL, iz, ix);
//           const int iglob = field.assembly_index_mapping(
//               field.index_mapping(ispec + lane, iz, ix),
//               static_cast<int>(MediumType));

//           if (iglob >= field.nglob) {
//             continue;
//           }

//           for (int icomp = 0; icomp < components; ++icomp) {
//             if constexpr (StoreDisplacement) {
//               element_field.displacement(iz, ix, icomp)[lane] =
//                   curr_field.field(iglob, icomp);
//             }
//             if constexpr (StoreVelocity) {
//               element_field.velocity(iz, ix, icomp)[lane] =
//                   curr_field.field_dot(iglob, icomp);
//             }
//             if constexpr (StoreAcceleration) {
//               element_field.acceleration(iz, ix, icomp)[lane] =
//                   curr_field.field_dot_dot(iglob, icomp);
//             }
//             if constexpr (StoreMassMatrix) {
//               element_field.mass_matrix(iz, ix, icomp)[lane] =
//                   curr_field.mass_inverse(iglob, icomp);
//             }
//           }
//         }
//       });

//   return;
// }

template <
    typename MemberType, specfem::wavefield::type WavefieldType,
    typename ViewType,
    typename std::enable_if_t<
        ViewType::isElementFieldType && !ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const MemberType &team, const int ispec,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &element_field) {

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

// template <
//     typename MemberType, specfem::wavefield::type WavefieldType,
//     typename ViewType,
//     typename std::enable_if_t<ViewType::isElementFieldType &&
//     ViewType::simd::using_simd,
//                      int> = 0>
// void load_on_host(
//     const MemberType &team, const int ispec,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     ViewType &element_field) {

//   constexpr static bool StoreDisplacement = ViewType::store_displacement;
//   constexpr static bool StoreVelocity = ViewType::store_velocity;
//   constexpr static bool StoreAcceleration = ViewType::store_acceleration;
//   constexpr static bool StoreMassMatrix = ViewType::store_mass_matrix;
//   constexpr static auto MediumType = ViewType::medium_tag;
//   constexpr static int components = ViewType::components;

//   constexpr static int NGLL = ViewType::ngll;

//   static_assert(
//       std::is_same_v<typename MemberType::execution_space,
//       Kokkos::HostSpace>, "Calling team must have a host execution space");

//   constexpr auto& curr_field = [&]() constexp
//       -> specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
//                                             MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
//         for (int lane = 0; lane < ViewType::simd::size(); ++lane) {
//           if (ispec + lane >= field.nspec) {
//             continue;
//           }

//           int iz, ix;
//           sub2ind(xz, NGLL, iz, ix);
//           const int iglob = field.h_assembly_index_mapping(
//               field.h_index_mapping(ispec + lane, iz, ix),
//               static_cast<int>(MediumType));

//           if (iglob >= field.nglob) {
//             continue;
//           }

//           for (int icomp = 0; icomp < components; ++icomp) {
//             if constexpr (StoreDisplacement) {
//               element_field.displacement(iz, ix, icomp)[lane] =
//                   curr_field.h_field(iglob, icomp);
//             }
//             if constexpr (StoreVelocity) {
//               element_field.velocity(iz, ix, icomp)[lane] =
//                   curr_field.h_field_dot(iglob, icomp);
//             }
//             if constexpr (StoreAcceleration) {
//               element_field.acceleration(iz, ix, icomp)[lane] =
//                   curr_field.h_field_dot_dot(iglob, icomp);
//             }
//             if constexpr (StoreMassMatrix) {
//               element_field.mass_matrix(iz, ix, icomp)[lane] =
//                   curr_field.h_mass_inverse(iglob, icomp);
//             }
//           }
//         }
//       });

//   return;
// }

template <
    typename MemberType, specfem::wavefield::type WavefieldType,
    typename IteratorType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && !ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const MemberType &team, const IteratorType &iterator,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &chunk_field) {

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
    typename MemberType, specfem::wavefield::type WavefieldType,
    typename IteratorType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && ViewType::simd::using_simd, int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const MemberType &team, const IteratorType &iterator,
               const specfem::compute::simulation_field<WavefieldType> &field,
               ViewType &chunk_field) {

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
    typename MemberType, specfem::wavefield::type WavefieldType,
    typename IteratorType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && !ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &chunk_field) {

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
    typename MemberType, specfem::wavefield::type WavefieldType,
    typename IteratorType, typename ViewType,
    typename std::enable_if_t<
        ViewType::isChunkFieldType && ViewType::simd::using_simd, int> = 0>
void load_on_host(
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::simulation_field<WavefieldType> &field,
    ViewType &chunk_field) {

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

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void load_on_device(
//     const int iglob,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     specfem::point::field<specfem::dimension::type::dim2, MediumType,
//                           StoreDisplacement, StoreVelocity,
//                           StoreAcceleration, StoreMassMatrix> &point_field) {
//   constexpr auto& curr_field = [&]() constexp
//       -> specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
//                                             MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   if constexpr (StoreDisplacement) {
//     point_field.displacement =
//         Kokkos::subview(curr_field.field, iglob, Kokkos::ALL);
//   }
//   if constexpr (StoreVelocity) {
//     point_field.velocity =
//         Kokkos::subview(curr_field.field_dot, iglob, Kokkos::ALL);
//   }
//   if constexpr (StoreAcceleration) {
//     point_field.acceleration =
//         Kokkos::subview(curr_field.field_dot_dot, iglob, Kokkos::ALL);
//   }

//   if constexpr (StoreMassMatrix) {
//     point_field.mass_matrix =
//         Kokkos::subview(curr_field.mass_inverse, iglob, Kokkos::ALL);
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void load_on_host(
//     const int iglob,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     specfem::point::field<specfem::dimension::type::dim2, MediumType,
//                           StoreDisplacement, StoreVelocity,
//                           StoreAcceleration, StoreMassMatrix> &point_field) {

//   constexpr auto& curr_field = [&]() constexp
//       -> specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
//                                             MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   if constexpr (StoreDisplacement) {
//     point_field.displacement =
//         Kokkos::subview(curr_field.h_field, iglob, Kokkos::ALL);
//   }

//   if constexpr (StoreVelocity) {
//     point_field.velocity =
//         Kokkos::subview(curr_field.h_field_dot, iglob, Kokkos::ALL);
//   }

//   if constexpr (StoreAcceleration) {
//     point_field.acceleration =
//         Kokkos::subview(curr_field.h_field_dot_dot, iglob, Kokkos::ALL);
//   }

//   if constexpr (StoreMassMatrix) {
//     point_field.mass_matrix =
//         Kokkos::subview(curr_field.h_mass_inverse, iglob, Kokkos::ALL);
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void load_on_device(
//     const specfem::point::index &index,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     specfem::point::field<specfem::dimension::type::dim2, MediumType,
//                           StoreDisplacement, StoreVelocity,
//                           StoreAcceleration, StoreMassMatrix> &point_field) {
//   const int iglob_l = field.index_mapping(index.ispec, index.iz, index.ix);
//   const int iglob =
//       field.assembly_index_mapping(iglob_l, static_cast<int>(MediumType));
//   load_on_device(iglob, field, point_field);
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void load_on_host(
//     const specfem::point::index &index,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     specfem::point::field<specfem::dimension::type::dim2, MediumType,
//                           StoreDisplacement, StoreVelocity,
//                           StoreAcceleration, StoreMassMatrix> &point_field) {
//   const int iglob = field.h_assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));
//   load_on_host(iglob, field, point_field);
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void store_on_device(
//     const int iglob,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   auto curr_field = [&]() -> specfem::compute::impl::field_impl<
//                               specfem::dimension::type::dim2, MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   for (int icomp = 0; icomp < components; ++icomp) {
//     if constexpr (StoreDisplacement) {
//       curr_field.field(iglob, icomp) = point_field.displacement[icomp];
//     }
//     if constexpr (StoreVelocity) {
//       curr_field.field_dot(iglob, icomp) = point_field.velocity[icomp];
//     }
//     if constexpr (StoreAcceleration) {
//       curr_field.field_dot_dot(iglob, icomp) =
//       point_field.acceleration[icomp];
//     }
//     if constexpr (StoreMassMatrix) {
//       curr_field.mass_inverse(iglob, icomp) = point_field.mass_matrix[icomp];
//     }
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void store_on_host(
//     const int iglob,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   auto curr_field = [&]() -> specfem::compute::impl::field_impl<
//                               specfem::dimension::type::dim2, MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   for (int icomp = 0; icomp < components; ++icomp) {
//     if constexpr (StoreDisplacement) {
//       curr_field.h_field(iglob, icomp) = point_field.displacement[icomp];
//     }
//     if constexpr (StoreVelocity) {
//       curr_field.h_field_dot(iglob, icomp) = point_field.velocity[icomp];
//     }
//     if constexpr (StoreAcceleration) {
//       curr_field.h_field_dot_dot(iglob, icomp) =
//           point_field.acceleration[icomp];
//     }
//     if constexpr (StoreMassMatrix) {
//       curr_field.h_mass_inverse(iglob, icomp) =
//       point_field.mass_matrix[icomp];
//     }
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void store_on_device(
//     const specfem::point::index &index,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   const int iglob = field.assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));

//   store_on_device(iglob, point_field, field);
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void store_on_host(
//     const specfem::point::index &index,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   const int iglob = field.h_assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));

//   store_on_host(iglob, point_field, field);
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void add_on_device(
//     const int iglob,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   auto curr_field = [&]() -> specfem::compute::impl::field_impl<
//                               specfem::dimension::type::dim2, MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   for (int icomp = 0; icomp < components; ++icomp) {
//     if constexpr (StoreDisplacement) {
//       curr_field.field(iglob, icomp) += point_field.displacement[icomp];
//     }
//     if constexpr (StoreVelocity) {
//       curr_field.field_dot(iglob, icomp) += point_field.velocity[icomp];
//     }
//     if constexpr (StoreAcceleration) {
//       curr_field.field_dot_dot(iglob, icomp) +=
//       point_field.acceleration[icomp];
//     }
//     if constexpr (StoreMassMatrix) {
//       curr_field.mass_inverse(iglob, icomp) +=
//       point_field.mass_matrix[icomp];
//     }
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void add_on_host(
//     const int iglob,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   auto curr_field = [&]() -> specfem::compute::impl::field_impl<
//                               specfem::dimension::type::dim2, MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   for (int icomp = 0; icomp < components; ++icomp) {
//     if constexpr (StoreDisplacement) {
//       curr_field.h_field(iglob, icomp) += point_field.displacement[icomp];
//     }
//     if constexpr (StoreVelocity) {
//       curr_field.h_field_dot(iglob, icomp) += point_field.velocity[icomp];
//     }
//     if constexpr (StoreAcceleration) {
//       curr_field.h_field_dot_dot(iglob, icomp) +=
//           point_field.acceleration[icomp];
//     }
//     if constexpr (StoreMassMatrix) {
//       curr_field.h_mass_inverse(iglob, icomp) +=
//       point_field.mass_matrix[icomp];
//     }
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void add_on_device(
//     const specfem::point::index &index,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   const int iglob = field.assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));

//   add_on_device(iglob, point_field, field);
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void add_on_host(
//     const specfem::point::index &index,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   const int iglob = field.h_assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));

//   add_on_host(iglob, point_field, field);

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void atomic_add_on_device(
//     const int iglob,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   auto curr_field = [&]() -> specfem::compute::impl::field_impl<
//                               specfem::dimension::type::dim2, MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   for (int icomp = 0; icomp < components; ++icomp) {
//     if constexpr (StoreDisplacement) {
//       Kokkos::atomic_add(&curr_field.field(iglob, icomp),
//                          point_field.displacement[icomp]);
//     }
//     if constexpr (StoreVelocity) {
//       Kokkos::atomic_add(&curr_field.field_dot(iglob, icomp),
//                          point_field.velocity[icomp]);
//     }
//     if constexpr (StoreAcceleration) {
//       Kokkos::atomic_add(&curr_field.field_dot_dot(iglob, icomp),
//                          point_field.acceleration[icomp]);
//     }
//     if constexpr (StoreMassMatrix) {
//       Kokkos::atomic_add(&curr_field.mass_inverse(iglob, icomp),
//                          point_field.mass_matrix[icomp]);
//     }
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void atomic_add_on_host(
//     const int iglob,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumType>::components;

//   auto curr_field = [&]() -> specfem::compute::impl::field_impl<
//                               specfem::dimension::type::dim2, MediumType> {
//     if constexpr (MediumType == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumType ==
//     specfem::element::medium_tag::acoustic) {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   for (int icomp = 0; icomp < components; ++icomp) {
//     if constexpr (StoreDisplacement) {
//       Kokkos::atomic_add(&curr_field.h_field(iglob, icomp),
//                          point_field.displacement[icomp]);
//     }
//     if constexpr (StoreVelocity) {
//       Kokkos::atomic_add(&curr_field.h_field_dot(iglob, icomp),
//                          point_field.velocity[icomp]);
//     }
//     if constexpr (StoreAcceleration) {
//       Kokkos::atomic_add(&curr_field.h_field_dot_dot(iglob, icomp),
//                          point_field.acceleration[icomp]);
//     }
//     if constexpr (StoreMassMatrix) {
//       Kokkos::atomic_add(&curr_field.h_mass_inverse(iglob, icomp),
//                          point_field.mass_matrix[icomp]);
//     }
//   }

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// NOINLINE KOKKOS_FUNCTION void atomic_add_on_device(
//     const specfem::point::index &index,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   const int iglob = field.assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));

//   atomic_add_on_device(iglob, point_field, field);
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::element::medium_tag MediumType, bool StoreDisplacement,
//           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
// void atomic_add_on_host(
//     const specfem::point::index &index,
//     const specfem::point::field<
//         specfem::dimension::type::dim2, MediumType, StoreDisplacement,
//         StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
//     const specfem::compute::simulation_field<WavefieldType> &field) {

//   const int iglob = field.h_assembly_index_mapping(
//       field.index_mapping(index.ispec, index.iz, index.ix),
//       static_cast<int>(MediumType));

//   atomic_add_on_host(iglob, point_field, field);
// }

// template <specfem::wavefield::type WavefieldType, typename MemberType,
//           typename FieldType,
//           std::enable_if_t<Kokkos::SpaceAccessibility<
//                                typename MemberType::execution_space,
//                                typename FieldType::memory_space>::accessible,
//                            bool> = true,
//           std::enable_if_t<FieldType::dimension_type ==
//                                specfem::dimension::type::dim2,
//                            bool> = true>
// NOINLINE KOKKOS_FUNCTION void
// load_on_device(const MemberType &team, const int ispec,
//                const specfem::compute::simulation_field<WavefieldType>
//                &field, FieldType &element_field) {

//   constexpr auto MediumTag = FieldType::medium_tag;
//   constexpr auto DimensionType = FieldType::dimension_type;
//   constexpr int NGLL = FieldType::ngll;
//   constexpr bool StoreDisplacement = FieldType::store_displacement;
//   constexpr bool StoreVelocity = FieldType::store_velocity;
//   constexpr bool StoreAcceleration = FieldType::store_acceleration;
//   constexpr bool StoreMassMatrix = FieldType::store_mass_matrix;

//   constexpr int components =
//       specfem::medium::medium<DimensionType, MediumTag>::components;

//   static_assert(std::is_same_v<typename MemberType::execution_space,
//                                specfem::kokkos::DevExecSpace>,
//                 "This function should only be called with device execution "
//                 "space");

//   constexpr auto& curr_field = [&]() constexp ->
//   specfem::compute::impl::field_impl<
//                                     specfem::dimension::type::dim2,
//                                     MediumTag> {
//     if constexpr (MediumTag == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumTag == specfem::element::medium_tag::acoustic)
//     {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
//         int iz, ix;
//         sub2ind(xz, NGLL, iz, ix);
//         const int iglob = field.assembly_index_mapping(
//             field.index_mapping(ispec, iz, ix), static_cast<int>(MediumTag));

//         for (int icomp = 0; icomp < components; ++icomp) {
//           if constexpr (StoreDisplacement) {
//             element_field.displacement(iz, ix, icomp) =
//                 curr_field.field(iglob, icomp);
//           }
//           if constexpr (StoreVelocity) {
//             element_field.velocity(iz, ix, icomp) =
//                 curr_field.field_dot(iglob, icomp);
//           }
//           if constexpr (StoreAcceleration) {
//             element_field.acceleration(iz, ix, icomp) =
//                 curr_field.field_dot_dot(iglob, icomp);
//           }
//           if constexpr (StoreMassMatrix) {
//             element_field.mass_matrix(iz, ix, icomp) =
//                 curr_field.mass_inverse(iglob, icomp);
//           }
//         }
//       });

//   return;
// }

// template <specfem::wavefield::type WavefieldType, typename MemberType,
//           typename FieldType,
//           std::enable_if_t<Kokkos::SpaceAccessibility<
//                                typename MemberType::execution_space,
//                                typename FieldType::memory_space>::accessible,
//                            bool> = true,
//           std::enable_if_t<FieldType::dimension_type ==
//                                specfem::dimension::type::dim2,
//                            bool> = true>
// void load_on_host(
//     const MemberType &team, const int ispec,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     FieldType &element_field) {

//   constexpr auto MediumTag = FieldType::medium_tag;
//   constexpr auto DimensionType = FieldType::dimension_type;
//   constexpr int NGLL = FieldType::ngll;
//   constexpr bool StoreDisplacement = FieldType::store_displacement;
//   constexpr bool StoreVelocity = FieldType::store_velocity;
//   constexpr bool StoreAcceleration = FieldType::store_acceleration;
//   constexpr bool StoreMassMatrix = FieldType::store_mass_matrix;

//   constexpr int components =
//       specfem::medium::medium<DimensionType, MediumTag>::components;

//   static_assert(
//       std::is_same_v<typename MemberType::execution_space,
//       Kokkos::HostSpace>, "This function should only be called with host
//       execution space");

//   constexpr auto& curr_field = [&]() constexp ->
//   specfem::compute::impl::field_impl<
//                                     specfem::dimension::type::dim2,
//                                     MediumTag> {
//     if constexpr (MediumTag == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumTag == specfem::element::medium_tag::acoustic)
//     {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
//         int iz, ix;
//         sub2ind(xz, NGLL, iz, ix);
//         const int iglob = field.h_assembly_index_mapping(
//             field.h_index_mapping(ispec, iz, ix),
//             static_cast<int>(MediumTag));

//         for (int icomp = 0; icomp < components; ++icomp) {
//           if constexpr (StoreDisplacement) {
//             element_field.displacement(iz, ix, icomp) =
//                 curr_field.h_field(iglob, icomp);
//           }
//           if constexpr (StoreVelocity) {
//             element_field.velocity(iz, ix, icomp) =
//                 curr_field.h_field_dot(iglob, icomp);
//           }
//           if constexpr (StoreAcceleration) {
//             element_field.acceleration(iz, ix, icomp) =
//                 curr_field.h_field_dot_dot(iglob, icomp);
//           }
//           if constexpr (StoreMassMatrix) {
//             element_field.mass_matrix(iz, ix, icomp) =
//                 curr_field.h_mass_inverse(iglob, icomp);
//           }
//         }
//       });

//   return;
// }

// template <specfem::wavefield::type WavefieldType, typename MemberType,
//           typename ChunkField,
//           std::enable_if_t<Kokkos::SpaceAccessibility<
//                                typename MemberType::execution_space,
//                                typename
//                                ChunkField::memory_space>::accessible,
//                            bool> = true,
//           std::enable_if_t<ChunkField::dimension_type ==
//                                specfem::dimension::type::dim2,
//                            bool> = true>
// NOINLINE KOKKOS_FUNCTION void load_on_device(
//     const MemberType &team,
//     const Kokkos::View<int *,
//                        typename MemberType::execution_space::memory_space>
//         &element_indices,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     ChunkField &element_field) {

//   constexpr auto MediumTag = ChunkField::medium_tag;
//   constexpr auto DimensionType = ChunkField::dimension_type;
//   constexpr int NumElements = ChunkField::num_elements;
//   constexpr int NGLL = ChunkField::ngll;
//   constexpr bool StoreDisplacement = ChunkField::store_displacement;
//   constexpr bool StoreVelocity = ChunkField::store_velocity;
//   constexpr bool StoreAcceleration = ChunkField::store_acceleration;
//   constexpr bool StoreMassMatrix = ChunkField::store_mass_matrix;

//   constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumTag>::components;

//   static_assert(std::is_same_v<typename MemberType::execution_space,
//                                specfem::kokkos::DevExecSpace>,
//                 "This function should only be called with device execution "
//                 "space");

//   const int nelements = element_indices.extent(0);

//   // DEVICE_ASSERT(nelements <= NumElements,
//   //        "Chunk element doesnt contain enough space for all elements");

//   constexpr auto& curr_field = [&]() constexp ->
//   specfem::compute::impl::field_impl<
//                                     specfem::dimension::type::dim2,
//                                     MediumTag> {
//     if constexpr (MediumTag == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumTag == specfem::element::medium_tag::acoustic)
//     {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, nelements * NGLL * NGLL),
//       [&](const int &ixz) {
//         const int ielement = ixz % nelements;
//         const int xz = ixz / nelements;
//         int iz, ix;
//         sub2ind(xz, NGLL, iz, ix);

//         const int ispec = element_indices(ielement);
//         const int iglob = field.assembly_index_mapping(
//             field.index_mapping(ispec, iz, ix), static_cast<int>(MediumTag));
//         for (int icomp = 0; icomp < components; ++icomp) {
//           if constexpr (StoreDisplacement) {
//             element_field.displacement(ielement, iz, ix, icomp) =
//                 curr_field.field(iglob, icomp);
//           }
//           if constexpr (StoreVelocity) {
//             element_field.velocity(ielement, iz, ix, icomp) =
//                 curr_field.field_dot(iglob, icomp);
//           }
//           if constexpr (StoreAcceleration) {
//             element_field.acceleration(ielement, iz, ix, icomp) =
//                 curr_field.field_dot_dot(iglob, icomp);
//           }
//           if constexpr (StoreMassMatrix) {
//             element_field.mass_matrix(ielement, iz, ix, icomp) =
//                 curr_field.mass_inverse(iglob, icomp);
//           }
//         }
//       });

//   return;
// }

// template <specfem::wavefield::type WavefieldType, typename MemberType,
//           typename ChunkField,
//           std::enable_if_t<Kokkos::SpaceAccessibility<
//                                typename MemberType::execution_space,
//                                typename
//                                ChunkField::memory_space>::accessible,
//                            bool> = true,
//           std::enable_if_t<ChunkField::dimension_type ==
//                                specfem::dimension::type::dim2,
//                            bool> = true>
// void load_on_host(
//     const MemberType &team,
//     const Kokkos::View<int *,
//                        typename MemberType::execution_space::memory_space>
//         &element_indices,
//     const specfem::compute::simulation_field<WavefieldType> &field,
//     ChunkField &element_field) {

//   constexpr auto MediumTag = ChunkField::medium_tag;
//   constexpr auto DimensionType = ChunkField::dimension_type;
//   constexpr int NumElements = ChunkField::num_elements;
//   constexpr int NGLL = ChunkField::ngll;
//   constexpr bool StoreDisplacement = ChunkField::store_displacement;
//   constexpr bool StoreVelocity = ChunkField::store_velocity;
//   constexpr bool StoreAcceleration = ChunkField::store_acceleration;
//   constexpr bool StoreMassMatrix = ChunkField::store_mass_matrix;

//   constexpr int components =
//       specfem::medium::medium<DimensionType, MediumTag>::components;

//   static_assert(
//       std::is_same_v<typename MemberType::execution_space,
//       Kokkos::HostSpace>, "This function should only be called with host
//       execution space");

//   const int nelements = element_indices.extent(0);

//   ASSERT(nelements <= NumElements,
//          "Chunk element doesnt contain enough space for all elements");

//   constexpr auto& curr_field = [&]() constexp ->
//   specfem::compute::impl::field_impl<
//                                     specfem::dimension::type::dim2,
//                                     MediumTag> {
//     if constexpr (MediumTag == specfem::element::medium_tag::elastic) {
//       return field.elastic;
//     } else if constexpr (MediumTag == specfem::element::medium_tag::acoustic)
//     {
//       return field.acoustic;
//     } else {
//       static_assert("medium type not supported");
//     }
//   }();

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, nelements * NGLL * NGLL),
//       [&](const int &ixz) {
//         const int ielement = ixz % nelements;
//         const int xz = ixz / nelements;
//         int iz, ix;
//         sub2ind(xz, NGLL, iz, ix);

//         const int ispec = element_indices(ielement);

//         const int iglob = field.h_assembly_index_mapping(
//             field.h_index_mapping(ispec, iz, ix),
//             static_cast<int>(MediumTag));
//         for (int icomp = 0; icomp < components; ++icomp) {
//           if constexpr (StoreDisplacement) {
//             element_field.displacement(ielement, iz, ix, icomp) =
//                 curr_field.h_field(iglob, icomp);
//           }
//           if constexpr (StoreVelocity) {
//             element_field.velocity(ielement, iz, ix, icomp) =
//                 curr_field.h_field_dot(iglob, icomp);
//           }
//           if constexpr (StoreAcceleration) {
//             element_field.acceleration(ielement, iz, ix, icomp) =
//                 curr_field.h_field_dot_dot(iglob, icomp);
//           }
//           if constexpr (StoreMassMatrix) {
//             element_field.mass_matrix(ielement, iz, ix, icomp) =
//                 curr_field.h_mass_inverse(iglob, icomp);
//           }
//         }
//       });

//   return;
// }

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_SIMULATION_FIELD_HPP_ */
