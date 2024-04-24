#ifndef _COMPUTE_FIELDS_SIMULATION_FIELD_HPP_
#define _COMPUTE_FIELDS_SIMULATION_FIELD_HPP_

#include "compute/fields/impl/field_impl.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
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

  template <specfem::sync::kind sync> void sync_fields() {
    elastic.sync_fields<sync>();
    acoustic.sync_fields<sync>();
  }

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

  int nglob = 0;
  int nspec;
  int ngllz;
  int ngllx;
  specfem::kokkos::DeviceView3d<int> index_mapping;
  specfem::kokkos::HostMirror3d<int> h_index_mapping;
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

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void
load_on_device(const int iglob,
               const specfem::compute::simulation_field<WavefieldType> &field,
               specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                     StoreDisplacement, StoreVelocity,
                                     StoreAcceleration> &point_field) {
  const auto curr_field = [&]()
      -> specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                            MediumType> {
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

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void load_on_host(
    const int iglob,
    const specfem::compute::simulation_field<WavefieldType> &field,
    specfem::point::field<specfem::dimension::type::dim2, MediumType,
                          StoreDisplacement, StoreVelocity, StoreAcceleration>
        &point_field) {

  const auto curr_field = [&]()
      -> specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                            MediumType> {
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

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void
load_on_device(const specfem::point::index &index,
               const specfem::compute::simulation_field<WavefieldType> &field,
               specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                     StoreDisplacement, StoreVelocity,
                                     StoreAcceleration> &point_field) {

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));
  load_on_device(iglob, field, point_field);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void load_on_host(
    const specfem::point::index &index,
    const specfem::compute::simulation_field<WavefieldType> &field,
    specfem::point::field<specfem::dimension::type::dim2, MediumType,
                          StoreDisplacement, StoreVelocity, StoreAcceleration>
        &point_field) {

  const int iglob = h_assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));
  load_on_host(iglob, field, point_field);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void store_on_device(
    const int iglob,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::field_impl<
                              specfem::dimension::type::dim2, MediumType> {
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
      curr_field.field(iglob, icomp) = point_field.displacement[icomp];
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.field_dot(iglob, icomp) = point_field.velocity[icomp];
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.field_dot_dot(iglob, icomp) = point_field.acceleration[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void store_on_host(
    const int iglob,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::field_impl<
                              specfem::dimension::type::dim2, MediumType> {
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
      curr_field.h_field(iglob, icomp) = point_field.displacement[icomp];
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.h_field_dot(iglob, icomp) = point_field.velocity[icomp];
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.h_field_dot_dot(iglob, icomp) =
          point_field.acceleration[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void store_on_device(
    const specfem::point::index &index,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  store_on_device(iglob, point_field, field);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void store_on_host(
    const specfem::point::index &index,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  const int iglob = field.h_assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  store_on_host(iglob, point_field, field);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void add_on_device(
    const int iglob,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::field_impl<
                              specfem::dimension::type::dim2, MediumType> {
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
      curr_field.field(iglob, icomp) += point_field.displacement[icomp];
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.field_dot(iglob, icomp) += point_field.velocity[icomp];
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.field_dot_dot(iglob, icomp) += point_field.acceleration[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void add_on_host(
    const int iglob,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::field_impl<
                              specfem::dimension::type::dim2, MediumType> {
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
      curr_field.h_field(iglob, icomp) += point_field.displacement[icomp];
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.h_field_dot(iglob, icomp) += point_field.velocity[icomp];
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      curr_field.h_field_dot_dot(iglob, icomp) +=
          point_field.acceleration[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void add_on_device(
    const specfem::point::index &index,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  add_on_device(iglob, point_field, field);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void add_on_host(
    const specfem::point::index &index,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  const int iglob = field.h_assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  add_on_host(iglob, point_field, field);

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void atomic_add_on_device(
    const int iglob,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::field_impl<
                              specfem::dimension::type::dim2, MediumType> {
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
      Kokkos::atomic_add(&curr_field.field(iglob, icomp),
                         point_field.displacement[icomp]);
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::atomic_add(&curr_field.field_dot(iglob, icomp),
                         point_field.velocity[icomp]);
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::atomic_add(&curr_field.field_dot_dot(iglob, icomp),
                         point_field.acceleration[icomp]);
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void atomic_add_on_host(
    const int iglob,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::field_impl<
                              specfem::dimension::type::dim2, MediumType> {
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
      Kokkos::atomic_add(&curr_field.h_field(iglob, icomp),
                         point_field.displacement[icomp]);
    }
  }

  if constexpr (StoreVelocity) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::atomic_add(&curr_field.h_field_dot(iglob, icomp),
                         point_field.velocity[icomp]);
    }
  }

  if constexpr (StoreAcceleration) {
    for (int icomp = 0; icomp < components; ++icomp) {
      Kokkos::atomic_add(&curr_field.h_field_dot_dot(iglob, icomp),
                         point_field.acceleration[icomp]);
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
KOKKOS_FUNCTION void atomic_add_on_device(
    const specfem::point::index &index,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  const int iglob = field.assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  atomic_add_on_device(iglob, point_field, field);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration>
void atomic_add_on_host(
    const specfem::point::index &index,
    const specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                StoreDisplacement, StoreVelocity,
                                StoreAcceleration> &point_field,
    const specfem::compute::simulation_field<WavefieldType> &field) {

  const int iglob = field.h_assembly_index_mapping(
      field.index_mapping(index.ispec, index.iz, index.ix),
      static_cast<int>(MediumType));

  atomic_add_on_host(iglob, point_field, field);
}

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_SIMULATION_FIELD_HPP_ */
