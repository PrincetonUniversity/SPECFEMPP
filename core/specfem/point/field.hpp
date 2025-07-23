#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

namespace impl {
template <typename ViewType, bool StoreDisplacement> struct Displacement {
  ViewType displacement; ///< Displacement at the quadrature point. Defined when
                         ///< StoreDisplacement is true.

  KOKKOS_FUNCTION
  Displacement() = default;

  KOKKOS_FUNCTION
  Displacement(const ViewType displacement) : displacement(displacement) {}
};

template <typename ViewType> struct Displacement<ViewType, false> {};

template <typename ViewType, bool StoreVelocity> struct Velocity {
  ViewType velocity; ///< Velocity at the quadrature point. Defined when
                     ///< StoreVelocity is true.

  KOKKOS_FUNCTION
  Velocity() = default;

  KOKKOS_FUNCTION
  Velocity(const ViewType velocity) : velocity(velocity) {}
};

template <typename ViewType> struct Velocity<ViewType, false> {};

template <typename ViewType, bool StoreAcceleration> struct Acceleration {
  ViewType acceleration; ///< Acceleration at the quadrature point. Defined when
                         ///< StoreAcceleration is true.

  KOKKOS_FUNCTION
  Acceleration() = default;

  KOKKOS_FUNCTION
  Acceleration(const ViewType acceleration) : acceleration(acceleration) {}
};

template <typename ViewType> struct Acceleration<ViewType, false> {};

template <typename ViewType, bool StoreMassMatrix> struct MassMatrix {
  ViewType mass_matrix; ///< Mass matrix at the quadrature point. Defined when
                        ///< StoreMassMatrix is true.

  KOKKOS_FUNCTION
  MassMatrix() = default;

  KOKKOS_FUNCTION
  MassMatrix(const ViewType mass_matrix) : mass_matrix(mass_matrix) {}

  /**
   * @brief Invert mass matrix
   *
   * This function is enabled when StoreMassMatrix is true.
   *
   * @return ViewType
   */
  KOKKOS_FUNCTION
  ViewType invert_mass_matrix() const {
    ViewType result;
    for (int i = 0; i < ViewType::components; ++i) {
      result(i) = static_cast<type_real>(1.0) / mass_matrix(i);
    }
    return result;
  }
};

template <typename ViewType> struct MassMatrix<ViewType, false> {};

template <typename ViewType, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix>
struct FieldTraits : public Displacement<ViewType, StoreDisplacement>,
                     public Velocity<ViewType, StoreVelocity>,
                     public Acceleration<ViewType, StoreAcceleration>,
                     public MassMatrix<ViewType, StoreMassMatrix> {

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

  KOKKOS_FUNCTION typename ViewType::value_type &
  operator()(const int i, std::true_type, std::false_type, std::false_type,
             std::false_type) {
    return this->displacement(i);
  }

  KOKKOS_FUNCTION typename ViewType::value_type &
  operator()(const int i, std::false_type, std::true_type, std::false_type,
             std::false_type) {
    return this->velocity(i);
  }

  KOKKOS_FUNCTION typename ViewType::value_type &
  operator()(const int i, std::false_type, std::false_type, std::true_type,
             std::false_type) {
    return this->acceleration(i);
  }

  KOKKOS_FUNCTION typename ViewType::value_type &
  operator()(const int i, std::false_type, std::false_type, std::false_type,
             std::true_type) {
    return this->mass_matrix(i);
  }

  KOKKOS_FUNCTION const typename ViewType::value_type &
  operator()(const int i, std::true_type, std::false_type, std::false_type,
             std::false_type) const {
    return this->displacement(i);
  }

  KOKKOS_FUNCTION const typename ViewType::value_type &
  operator()(const int i, std::false_type, std::true_type, std::false_type,
             std::false_type) const {
    return this->velocity(i);
  }

  KOKKOS_FUNCTION const typename ViewType::value_type &
  operator()(const int i, std::false_type, std::false_type, std::true_type,
             std::false_type) const {
    return this->acceleration(i);
  }

  KOKKOS_FUNCTION const typename ViewType::value_type &
  operator()(const int i, std::false_type, std::false_type, std::false_type,
             std::true_type) const {
    return this->mass_matrix(i);
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

  template <typename... Args>
  KOKKOS_FUNCTION FieldTraits(Args... args)
      : FieldTraits(ViewType(std::forward<Args>(args)...)) {}

  KOKKOS_FUNCTION typename ViewType::value_type &operator()(const int i) {
    return operator()(i, std::integral_constant<bool, StoreDisplacement>{},
                      std::integral_constant<bool, StoreVelocity>{},
                      std::integral_constant<bool, StoreAcceleration>{},
                      std::integral_constant<bool, StoreMassMatrix>{});
  }

  KOKKOS_FUNCTION const typename ViewType::value_type &
  operator()(const int i) const {
    return operator()(i, std::integral_constant<bool, StoreDisplacement>{},
                      std::integral_constant<bool, StoreVelocity>{},
                      std::integral_constant<bool, StoreAcceleration>{},
                      std::integral_constant<bool, StoreMassMatrix>{});
  }

  /**
   * @brief Divide acceleration by mass matrix
   *
   * This function is enabled when StoreAcceleration and StoreMassMatrix are
   * true.
   *
   * @return ViewType Acceleration divided by mass matrix
   */
  KOKKOS_FUNCTION
  ViewType divide_mass_matrix() const {
    return divide_mass_matrix(std::integral_constant<bool, StoreAcceleration>{},
                              std::integral_constant<bool, StoreMassMatrix>{});
  }

  KOKKOS_FUNCTION bool operator==(const FieldTraits &other) const {
    bool result = true;
    if constexpr (StoreDisplacement) {
      result = result && (this->displacement == other.displacement);
    }
    if constexpr (StoreVelocity) {
      result = result && (this->velocity == other.velocity);
    }
    if constexpr (StoreAcceleration) {
      result = result && (this->acceleration == other.acceleration);
    }
    if constexpr (StoreMassMatrix) {
      result = result && (this->mass_matrix == other.mass_matrix);
    }
    return result;
  }

  KOKKOS_FUNCTION
  bool operator!=(const FieldTraits &other) const { return !(*this == other); }
};

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct FieldAccessor
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::field, DimensionTag, UseSIMD> {
private:
  using base_type =
      specfem::data_access::Accessor<specfem::data_access::AccessorType::point,
                                     specfem::data_access::DataClassType::field,
                                     DimensionTag, UseSIMD>; ///< Base accessor
                                                             ///< type
public:
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type = typename base_type::template vector_type<
      type_real,
      specfem::element::attributes<DimensionTag, MediumTag>::components>;
};

} // namespace impl

/**
 * @brief Point field type to store displacement, velocity, acceleration and
 mass
 * matrix at a quadrature point
 *
 * @tparam DimensionTag Dimension of the element where the quadrature point is
 * located
 * @tparam MediumTag Medium type of the element where the quadrature point is
 * located
 * @tparam StoreDisplacement Store displacement at the quadrature point
 * @tparam StoreVelocity Store velocity at the quadrature point
 * @tparam StoreAcceleration Store acceleration at the quadrature point
 * @tparam StoreMassMatrix Store mass matrix at the quadrature point
 * @tparam UseSIMD Boolean to enable SIMD operations
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix,
          bool UseSIMD>
struct field : public impl::FieldAccessor<DimensionTag, MediumTag, UseSIMD>,
               public impl::FieldTraits<
                   typename impl::FieldAccessor<DimensionTag, MediumTag,
                                                UseSIMD>::value_type,
                   StoreDisplacement, StoreVelocity, StoreAcceleration,
                   StoreMassMatrix> {
private:
  using base_accessor = impl::FieldAccessor<DimensionTag, MediumTag, UseSIMD>;
  using base_traits =
      impl::FieldTraits<typename base_accessor::value_type, StoreDisplacement,
                        StoreVelocity, StoreAcceleration, StoreMassMatrix>;

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_accessor::simd;             ///< SIMD type
  using value_type = typename base_accessor::value_type; ///< Underlying
                                                         ///< datatype used
                                                         ///< to store the
                                                         ///< field
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int components =
      value_type::components; ///< Number of field components for specified
                              ///< medium
  constexpr static auto medium_tag =
      MediumTag; ///< Medium type of the element where the quadrature point is
                 ///< located

  constexpr static bool store_displacement =
      StoreDisplacement; ///< Store displacement at the quadrature point
  constexpr static bool store_velocity =
      StoreVelocity; ///< Store velocity at the quadrature point
  constexpr static bool store_acceleration =
      StoreAcceleration; ///< Store acceleration at the quadrature point
  constexpr static bool store_mass_matrix =
      StoreMassMatrix; ///< Store mass matrix at the quadrature point

  constexpr static bool isChunkFieldType =
      value_type::isChunkViewType; ///< Check if the field is a chunk field type
                                   ///< (false)
  constexpr static bool isElementFieldType =
      value_type::isElementViewType; ///< Check if the field is an element field
                                     ///< type (false)
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{
  using base_traits::base_traits; ///< Inherit constructors from traits type
  ///@}
};

} // namespace point
} // namespace specfem
