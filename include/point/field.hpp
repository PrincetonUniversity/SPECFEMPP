#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
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
struct ImplFieldTraits : public Displacement<ViewType, StoreDisplacement>,
                         public Velocity<ViewType, StoreVelocity>,
                         public Acceleration<ViewType, StoreAcceleration>,
                         public MassMatrix<ViewType, StoreMassMatrix> {

private:
  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType displacement, std::true_type, std::false_type,
                  std::false_type, std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType velocity, std::false_type, std::true_type,
                  std::false_type, std::false_type)
      : impl::Velocity<ViewType, StoreVelocity>(velocity) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType acceleration, std::false_type, std::false_type,
                  std::true_type, std::false_type)
      : impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType mass_matrix, std::false_type, std::false_type,
                  std::false_type, std::true_type)
      : impl::MassMatrix<ViewType, StoreMassMatrix>(mass_matrix) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType displacement, const ViewType velocity,
                  std::true_type, std::true_type, std::false_type,
                  std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement),
        impl::Velocity<ViewType, StoreVelocity>(velocity) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType displacement, const ViewType acceleration,
                  std::true_type, std::false_type, std::true_type,
                  std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(displacement),
        impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType velocity, const ViewType acceleration,
                  std::false_type, std::true_type, std::true_type,
                  std::false_type)
      : impl::Velocity<ViewType, StoreVelocity>(velocity),
        impl::Acceleration<ViewType, StoreAcceleration>(acceleration) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType displacement, const ViewType velocity,
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

public:
  KOKKOS_FUNCTION
  ImplFieldTraits() = default;

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType view)
      : ImplFieldTraits(view, std::integral_constant<bool, StoreDisplacement>{},
                        std::integral_constant<bool, StoreVelocity>{},
                        std::integral_constant<bool, StoreAcceleration>{},
                        std::integral_constant<bool, StoreMassMatrix>{}) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType view1, const ViewType view2)
      : ImplFieldTraits(view1, view2,
                        std::integral_constant<bool, StoreDisplacement>{},
                        std::integral_constant<bool, StoreVelocity>{},
                        std::integral_constant<bool, StoreAcceleration>{},
                        std::integral_constant<bool, StoreMassMatrix>{}) {}

  KOKKOS_FUNCTION
  ImplFieldTraits(const ViewType view1, const ViewType view2,
                  const ViewType view3)
      : ImplFieldTraits(view1, view2, view3,
                        std::integral_constant<bool, StoreDisplacement>{},
                        std::integral_constant<bool, StoreVelocity>{},
                        std::integral_constant<bool, StoreAcceleration>{},
                        std::integral_constant<bool, StoreMassMatrix>{}) {}

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
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix,
          bool UseSIMD>
struct FieldTraits
    : public ImplFieldTraits<specfem::datatype::ScalarPointViewType<
                                 type_real,
                                 specfem::element::attributes<
                                     DimensionType, MediumTag>::components(),
                                 UseSIMD>,
                             StoreDisplacement, StoreVelocity,
                             StoreAcceleration, StoreMassMatrix> {

  using ViewType = specfem::datatype::ScalarPointViewType<
      type_real,
      specfem::element::attributes<DimensionType, MediumTag>::components(),
      UseSIMD>;

public:
  KOKKOS_FUNCTION
  FieldTraits() = default;

  KOKKOS_FUNCTION
  FieldTraits(const ViewType view)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(view) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType view1, const ViewType view2)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(view1, view2) {}

  KOKKOS_FUNCTION
  FieldTraits(const ViewType view1, const ViewType view2, const ViewType view3)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(view1, view2,
                                                            view3) {}

  template <typename... Args>
  KOKKOS_FUNCTION FieldTraits(Args... args)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(
            ViewType(std::forward<Args>(args)...)) {}
};

} // namespace impl

/**
 * @brief Point field type to store displacement, velocity, acceleration and
 mass
 * matrix at a quadrature point
 *
 * @tparam DimensionType Dimension of the element where the quadrature point is
 * located
 * @tparam MediumType Medium type of the element where the quadrature point is
 * located
 * @tparam StoreDisplacement Store displacement at the quadrature point
 * @tparam StoreVelocity Store velocity at the quadrature point
 * @tparam StoreAcceleration Store acceleration at the quadrature point
 * @tparam StoreMassMatrix Store mass matrix at the quadrature point
 * @tparam UseSIMD Boolean to enable SIMD operations
 *
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix,
          bool UseSIMD>
struct field
    : public impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                               StoreVelocity, StoreAcceleration,
                               StoreMassMatrix, UseSIMD> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using ViewType = typename impl::FieldTraits<
      DimensionType, MediumType, StoreDisplacement, StoreVelocity,
      StoreAcceleration, StoreMassMatrix, UseSIMD>::ViewType; ///< Underlying
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
      ViewType::components; ///< Number of field components for specified medium
  constexpr static auto dimension = DimensionType; ///< Dimension of the element
                                                   ///< where the quadrature
                                                   ///< point is located
  constexpr static auto medium_tag =
      MediumType; ///< Medium type of the element where the quadrature point is
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
      ViewType::isChunkViewType; ///< Check if the field is a chunk field type
                                 ///< (false)
  constexpr static bool isPointFieldType =
      ViewType::isPointViewType; ///< Check if the field is a point field type
                                 ///< (true)
  constexpr static bool isElementFieldType =
      ViewType::isElementViewType; ///< Check if the field is an element field
                                   ///< type (false)
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  field() = default;

  /**
   * @brief Construct new point field object with single view type.
   *
   * This constructore is enabled when only one of StoreDisplacement,
   * StoreVelocity, StoreAcceleration, StoreMassMatrix is true.
   *
   * @param view View associated with field type defined by the template
   * parameters
   */
  KOKKOS_FUNCTION
  field(const ViewType view)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(view) {}

  /**
   * @brief Construct new point field object with two view types.
   *
   * This constructor is enabled when two of StoreDisplacement, StoreVelocity,
   * and StoreAcceleration are true.
   *
   * @param view1
   * displacement view
   * @code if ((StoreDisplacement && StoreVelocity == true) ||
   * (StoreDisplacement && StoreAcceleration == true)) @endcode.
   *  velocity view
   * if @code (StoreVelocity && StoreAcceleration == true) @endcode
   * @param view2
   * velocity view
   * if @code (StoreDisplacement && StoreVelocity == true) @endcode.
   * acceleration view
   * if @code (StoreDisplacement && StoreAcceleration == true) || StoreVelocity
   * && StoreAcceleration == true) @endcode
   */
  KOKKOS_FUNCTION
  field(const ViewType view1, const ViewType view2)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(view1, view2) {}

  /**
   * @brief Construct new point field object with three view types.
   *
   * This constructor is enabled when StoreDisplacement, StoreVelocity, and
   * StoreAcceleration are true.
   * @param view1 displacement view
   * @param view2 velocity view
   * @param view3 acceleration view
   */
  KOKKOS_FUNCTION
  field(const ViewType view1, const ViewType view2, const ViewType view3)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(view1, view2, view3) {}

  template <typename... Args>
  KOKKOS_FUNCTION field(Args... args)
      : impl::FieldTraits<DimensionType, MediumType, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(ViewType(std::forward<Args>(args)...)) {}
  ///@}
};

} // namespace point
} // namespace specfem
