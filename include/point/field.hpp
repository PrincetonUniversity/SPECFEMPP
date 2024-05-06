#ifndef _POINT_FIELD_HPP_
#define _POINT_FIELD_HPP_

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
struct field;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, true, true, true, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> displacement;
  specfem::kokkos::array_type<type_real, components> velocity;
  specfem::kokkos::array_type<type_real, components> acceleration;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> displacement,
        const specfem::kokkos::array_type<type_real, components> velocity,
        const specfem::kokkos::array_type<type_real, components> acceleration)
      : displacement(displacement), velocity(velocity),
        acceleration(acceleration) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, true, false, false, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> displacement;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> displacement)
      : displacement(displacement) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, false, true, false, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> velocity;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> velocity)
      : velocity(velocity) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, false, false, true, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> acceleration;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> acceleration)
      : acceleration(acceleration) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, true, true, false, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> displacement;
  specfem::kokkos::array_type<type_real, components> velocity;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> displacement,
        const specfem::kokkos::array_type<type_real, components> velocity)
      : displacement(displacement), velocity(velocity) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, true, false, true, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> displacement;
  specfem::kokkos::array_type<type_real, components> acceleration;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> displacement,
        const specfem::kokkos::array_type<type_real, components> acceleration)
      : displacement(displacement), acceleration(acceleration) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, false, true, true, false> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> velocity;
  specfem::kokkos::array_type<type_real, components> acceleration;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> velocity,
        const specfem::kokkos::array_type<type_real, components> acceleration)
      : velocity(velocity), acceleration(acceleration) {}
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, false, false, false, true> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> mass_matrix;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> mass_matrix)
      : mass_matrix(mass_matrix) {}

  KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, components>
  invert_mass_matrix() const {
    specfem::kokkos::array_type<type_real, components> result;
    for (int i = 0; i < components; i++) {
      result[i] = 1.0 / mass_matrix[i];
    }
    return result;
  }
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType>
struct field<DimensionType, MediumType, false, false, true, true> {
public:
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;

  specfem::kokkos::array_type<type_real, components> acceleration;
  specfem::kokkos::array_type<type_real, components> mass_matrix;

  field() = default;
  field(const specfem::kokkos::array_type<type_real, components> acceleration,
        const specfem::kokkos::array_type<type_real, components> mass_matrix)
      : acceleration(acceleration), mass_matrix(mass_matrix) {}

  KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, components>
  divide_mass_matrix() const {
    specfem::kokkos::array_type<type_real, components> result;
    for (int i = 0; i < components; ++i) {
      result[i] = acceleration[i] * mass_matrix[i];
    }
    return result;
  }
};

} // namespace point
} // namespace specfem

#endif
