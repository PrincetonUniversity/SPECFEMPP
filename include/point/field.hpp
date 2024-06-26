#ifndef _POINT_FIELD_HPP_
#define _POINT_FIELD_HPP_

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace point {

namespace impl {
template <typename ViewType, bool StoreDisplacement> struct Displacement;

template <typename ViewType> struct Displacement<ViewType, true> {
  ViewType displacement;

  KOKKOS_FUNCTION
  Displacement() = default;

  KOKKOS_FUNCTION
  Displacement(const ViewType displacement) : displacement(displacement) {}
};

template <typename ViewType> struct Displacement<ViewType, false> {};

template <typename ViewType, bool StoreVelocity> struct Velocity;

template <typename ViewType> struct Velocity<ViewType, true> {
  ViewType velocity;

  KOKKOS_FUNCTION
  Velocity() = default;

  KOKKOS_FUNCTION
  Velocity(const ViewType velocity) : velocity(velocity) {}
};

template <typename ViewType> struct Velocity<ViewType, false> {};

template <typename ViewType, bool StoreAcceleration> struct Acceleration;

template <typename ViewType> struct Acceleration<ViewType, true> {
  ViewType acceleration;

  KOKKOS_FUNCTION
  Acceleration() = default;

  KOKKOS_FUNCTION
  Acceleration(const ViewType acceleration) : acceleration(acceleration) {}
};

template <typename ViewType> struct Acceleration<ViewType, false> {};

template <typename ViewType, bool StoreMassMatrix> struct MassMatrix;

template <typename ViewType> struct MassMatrix<ViewType, true> {
  ViewType mass_matrix;

  KOKKOS_FUNCTION
  MassMatrix() = default;

  KOKKOS_FUNCTION
  MassMatrix(const ViewType mass_matrix) : mass_matrix(mass_matrix) {}

  KOKKOS_FUNCTION
  ViewType invert_mass_matrix() const {
    ViewType result;
    for (int i = 0; i < ViewType::components; ++i) {
      result(i) = 1.0 / mass_matrix(i);
    }
    return result;
  }
};

template <typename ViewType> struct MassMatrix<ViewType, false> {};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
struct FieldTraits
    : public Displacement<
          typename specfem::datatype::ScalarPointViewType<
              type_real,
              specfem::medium::medium<DimensionType, MediumTag>::components>,
          StoreDisplacement>,
      public Velocity<typename specfem::datatype::ScalarPointViewType<
                          type_real, specfem::medium::medium<
                                         DimensionType, MediumTag>::components>,
                      StoreVelocity>,
      public Acceleration<
          typename specfem::datatype::ScalarPointViewType<
              type_real,
              specfem::medium::medium<DimensionType, MediumTag>::components>,
          StoreAcceleration>,
      public MassMatrix<
          typename specfem::datatype::ScalarPointViewType<
              type_real,
              specfem::medium::medium<DimensionType, MediumTag>::components>,
          StoreMassMatrix> {

public:
  constexpr static int components =
      specfem::medium::medium<DimensionType, MediumTag>::components;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto dimension_type = DimensionType;
  constexpr static bool store_displacement = StoreDisplacement;
  constexpr static bool store_velocity = StoreVelocity;
  constexpr static bool store_acceleration = StoreAcceleration;
  constexpr static bool store_mass_matrix = StoreMassMatrix;

  using ViewType =
      typename specfem::datatype::ScalarPointViewType<type_real, components>;

  constexpr static bool isChunkFieldType = ViewType::isChunkViewType;
  constexpr static bool isPointFieldType = ViewType::isPointViewType;
  constexpr static int isElementFieldType = ViewType::isElementViewType;

private:
  KOKKOS_FUNCTION
  FieldTraits(const ViewType displacement, std::true_type, std::false_type,
              std::false_type, std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType velocity, std::false_type, std::true_type,
              std::false_type, std::false_type)
      : impl::Velocity<ViewType, StoreVelocity>(velocity) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType acceleration, std::false_type, std::false_type,
              std::true_type, std::false_type)
      : impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType mass_matrix, std::false_type, std::false_type,
              std::false_type, std::true_type)
      : impl::MassMatrix<ViewType, StoreMassMatrix>(mass_matrix) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType displacement, const ViewType velocity,
              std::true_type, std::true_type, std::false_type, std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement),
        impl::Velocity<ViewType, StoreVelocity>(velocity) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType displacement, const ViewType acceleration,
              std::true_type, std::false_type, std::true_type, std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement),
        impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType velocity, const ViewType acceleration,
              std::false_type, std::true_type, std::true_type, std::false_type)
      : impl::Velocity<ViewType, StoreVelocity>(velocity),
        impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType displacement, const ViewType velocity,
              const ViewType acceleration, std::true_type, std::true_type,
              std::true_type, std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement),
        impl::Velocity<ViewType, StoreVelocity>(velocity),
        impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  ViewType divide_mass_matrix(std::true_type, std::true_type) const {
    ViewType result;
    for (int i = 0; i < ViewType::components; ++i) {
      result(i) = this->acceleration(i) * this->mass_matrix(i);
    }
    return result;
  }

  KOKKOS_FUNCTION
  ViewType divide_mass_matrix(std::true_type, std::false_type) const {
    static_assert(StoreMassMatrix, "Mass matrix is not stored");
  }

  KOKKOS_FUNCTION
  ViewType divide_mass_matrix(std::false_type, std::true_type) const {
    static_assert(StoreAcceleration, "Acceleration is not stored");
  }

  KOKKOS_FUNCTION
  ViewType divide_mass_matrix(std::false_type, std::false_type) const {
    static_assert(StoreAcceleration && StoreMassMatrix,
                  "Acceleration and mass matrix are not stored");
  }

public:
  KOKKOS_FUNCTION
  FieldTraits() = default;

  KOKKOS_FUNCTION
  FieldTraits(const ViewType view)
      : FieldTraits(view, std::integral_constant<bool, StoreDisplacement>{},
                    std::integral_constant<bool, StoreVelocity>{},
                    std::integral_constant<bool, StoreAcceleration>{},
                    std::integral_constant<bool, StoreMassMatrix>{}) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType view1, const ViewType view2)
      : FieldTraits(view1, view2,
                    std::integral_constant<bool, StoreDisplacement>{},
                    std::integral_constant<bool, StoreVelocity>{},
                    std::integral_constant<bool, StoreAcceleration>{},
                    std::integral_constant<bool, StoreMassMatrix>{}) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType view1, const ViewType view2, const ViewType view3)
      : FieldTraits(view1, view2, view3,
                    std::integral_constant<bool, StoreDisplacement>{},
                    std::integral_constant<bool, StoreVelocity>{},
                    std::integral_constant<bool, StoreAcceleration>{},
                    std::integral_constant<bool, StoreMassMatrix>{}) {}

  KOKKOS_FUNCTION
  ViewType divide_mass_matrix() const {
    return divide_mass_matrix(std::integral_constant<bool, StoreAcceleration>{},
                              std::integral_constant<bool, StoreMassMatrix>{});
  }
};

} // namespace impl

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
struct field : public impl::FieldTraits<DimensionType, MediumType,
                                        StoreDisplacement, StoreVelocity,
                                        StoreAcceleration, StoreMassMatrix> {

  using ViewType =
      typename impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                                 StoreVelocity, StoreAcceleration,
                                 StoreMassMatrix>::ViewType;

  KOKKOS_FUNCTION
  field() = default;

  KOKKOS_FUNCTION
  field(const ViewType view)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix>(
            view) {}

  KOKKOS_FUNCTION
  field(const ViewType view1, const ViewType view2)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix>(
            view1, view2) {}

  KOKKOS_FUNCTION
  field(const ViewType view1, const ViewType view2, const ViewType view3)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix>(
            view1, view2, view3) {}

  KOKKOS_FUNCTION
  ViewType divide_mass_matrix() const {
    return impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                             StoreVelocity, StoreAcceleration,
                             StoreMassMatrix>::divide_mass_matrix();
  }
};

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, true, true, true, false>
//     : public impl::FieldTraits<DimensionType, MediumType, true, true, true,
//                                false> {

//   using ViewType = typename impl::FieldTraits<DimensionType, MediumType,
//   true,
//                                               true, true, false>::ViewType;

//   ViewType displacement;
//   ViewType velocity;
//   ViewType acceleration;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const ViewType displacement, const ViewType velocity,
//         const ViewType acceleration)
//       : displacement(displacement), velocity(velocity),
//         acceleration(acceleration) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, true, false, false, false> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> displacement;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components>
//   displacement)
//       : displacement(displacement) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, false, true, false, false> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> velocity;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components> velocity)
//       : velocity(velocity) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, false, false, true, false> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> acceleration;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components>
//   acceleration)
//       : acceleration(acceleration) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, true, true, false, false> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> displacement;
//   specfem::kokkos::array_type<type_real, components> velocity;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components>
//   displacement,
//         const specfem::kokkos::array_type<type_real, components> velocity)
//       : displacement(displacement), velocity(velocity) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, true, false, true, false> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> displacement;
//   specfem::kokkos::array_type<type_real, components> acceleration;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components>
//   displacement,
//         const specfem::kokkos::array_type<type_real, components>
//         acceleration)
//       : displacement(displacement), acceleration(acceleration) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, false, true, true, false> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> velocity;
//   specfem::kokkos::array_type<type_real, components> acceleration;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components> velocity,
//         const specfem::kokkos::array_type<type_real, components>
//         acceleration)
//       : velocity(velocity), acceleration(acceleration) {}
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, false, false, false, true> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> mass_matrix;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components> mass_matrix)
//       : mass_matrix(mass_matrix) {}

//   KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, components>
//   invert_mass_matrix() const {
//     specfem::kokkos::array_type<type_real, components> result;
//     for (int i = 0; i < components; i++) {
//       result[i] = 1.0 / mass_matrix[i];
//     }
//     return result;
//   }
// };

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumType>
// struct field<DimensionType, MediumType, false, false, true, true> {
// public:
//   static constexpr int components =
//       specfem::medium::medium<DimensionType, MediumType>::components;

//   specfem::kokkos::array_type<type_real, components> acceleration;
//   specfem::kokkos::array_type<type_real, components> mass_matrix;

//   KOKKOS_FUNCTION
//   field() = default;

//   KOKKOS_FUNCTION
//   field(const specfem::kokkos::array_type<type_real, components>
//   acceleration,
//         const specfem::kokkos::array_type<type_real, components> mass_matrix)
//       : acceleration(acceleration), mass_matrix(mass_matrix) {}

//   KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, components>
//   divide_mass_matrix() const {
//     specfem::kokkos::array_type<type_real, components> result;
//     for (int i = 0; i < components; ++i) {
//       result[i] = acceleration[i] * mass_matrix[i];
//     }
//     return result;
//   }
// };

} // namespace point
} // namespace specfem

#endif
