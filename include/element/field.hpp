#ifndef _ELEMENT_FIELD_HPP
#define _ELEMENT_FIELD_HPP

#include "datatypes/element_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace element {

namespace impl {

template <typename ViewType, bool StoreDisplacement> struct Displacement;

template <typename ViewType> struct Displacement<ViewType, true> {
  using view_type = ViewType;
  ViewType displacement;

  KOKKOS_FUNCTION Displacement() = default;

  KOKKOS_FUNCTION Displacement(const ViewType &displacement)
      : displacement(displacement) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Displacement(const ScratchMemorySpace &scratch_space)
      : displacement(scratch_space) {}
};

template <typename ViewType> struct Displacement<ViewType, false> {
  using view_type = ViewType;
};

template <typename ViewType, bool StoreVelocity> struct Velocity;

template <typename ViewType> struct Velocity<ViewType, true> {
  using view_type = ViewType;
  ViewType velocity;

  KOKKOS_FUNCTION Velocity() = default;

  KOKKOS_FUNCTION Velocity(const ViewType &velocity) : velocity(velocity) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Velocity(const ScratchMemorySpace &scratch_space)
      : velocity(scratch_space) {}
};

template <typename ViewType> struct Velocity<ViewType, false> {
  using view_type = ViewType;
};

template <typename ViewType, bool StoreAcceleration> struct Acceleration;

template <typename ViewType> struct Acceleration<ViewType, true> {
  using view_type = ViewType;
  ViewType acceleration;

  KOKKOS_FUNCTION Acceleration() = default;

  KOKKOS_FUNCTION Acceleration(const ViewType &acceleration)
      : acceleration(acceleration) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Acceleration(const ScratchMemorySpace &scratch_space)
      : acceleration(scratch_space) {}
};

template <typename ViewType> struct Acceleration<ViewType, false> {
  using view_type = ViewType;
};

template <typename ViewType, bool StoreMassMatrix> struct MassMatrix;

template <typename ViewType> struct MassMatrix<ViewType, true> {
  using view_type = ViewType;
  ViewType mass_matrix;

  KOKKOS_FUNCTION MassMatrix() = default;

  KOKKOS_FUNCTION MassMatrix(const ViewType &mass_matrix)
      : mass_matrix(mass_matrix) {}
};

template <typename ViewType> struct MassMatrix<ViewType, false> {
  using view_type = ViewType;
};

template <typename ViewType, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix>
struct ImplFieldTraits : Displacement<ViewType, StoreDisplacement>,
                         Velocity<ViewType, StoreVelocity>,
                         Acceleration<ViewType, StoreAcceleration>,
                         MassMatrix<ViewType, StoreMassMatrix> {
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

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::true_type,
                                  std::false_type, std::false_type,
                                  std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::false_type,
                                  std::true_type, std::false_type,
                                  std::false_type)
      : impl::Velocity<ViewType, StoreVelocity>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::false_type,
                                  std::false_type, std::true_type,
                                  std::false_type)
      : impl::Acceleration<ViewType, StoreAcceleration>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::false_type,
                                  std::false_type, std::false_type,
                                  std::true_type)
      : impl::MassMatrix<ViewType, StoreMassMatrix>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::true_type,
                                  std::true_type, std::false_type,
                                  std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(team.team_scratch(0)),
        impl::Velocity<ViewType, StoreVelocity>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::true_type,
                                  std::false_type, std::true_type,
                                  std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(team.team_scratch(0)),
        impl::Acceleration<ViewType, StoreAcceleration>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::false_type,
                                  std::true_type, std::true_type,
                                  std::false_type)
      : impl::Velocity<ViewType, StoreVelocity>(team.team_scratch(0)),
        impl::Acceleration<ViewType, StoreAcceleration>(team.team_scratch(0)) {}

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team, std::true_type,
                                  std::true_type, std::true_type,
                                  std::false_type)
      : impl::Displacement<ViewType, StoreDisplacement>(team.team_scratch(0)),
        impl::Velocity<ViewType, StoreVelocity>(team.team_scratch(0)),
        impl::Acceleration<ViewType, StoreAcceleration>(team.team_scratch(0)) {}

public:
  KOKKOS_FUNCTION
  ImplFieldTraits() = default;

  template <typename MemberType>
  KOKKOS_FUNCTION ImplFieldTraits(const MemberType &team)
      : ImplFieldTraits(team, std::integral_constant<bool, StoreDisplacement>{},
                        std::integral_constant<bool, StoreVelocity>{},
                        std::integral_constant<bool, StoreAcceleration>{},
                        std::integral_constant<bool, StoreMassMatrix>{}) {}

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

  static int shmem_size() {
    return (static_cast<int>(StoreDisplacement) +
            static_cast<int>(StoreVelocity) +
            static_cast<int>(StoreAcceleration) +
            static_cast<int>(StoreMassMatrix)) *
           ViewType::shmem_size();
  }
};

template <int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix, bool UseSIMD>
struct FieldTraits
    : ImplFieldTraits<
          specfem::datatype::ScalarElementViewType<
              type_real, NGLL,
              specfem::element::attributes<DimensionTag, MediumTag>::components,
              MemorySpace, MemoryTraits, UseSIMD>,
          StoreDisplacement, StoreVelocity, StoreAcceleration,
          StoreMassMatrix> {
public:
  using ViewType = specfem::datatype::ScalarElementViewType<
      type_real, NGLL,
      specfem::element::attributes<DimensionTag, MediumTag>::components,
      MemorySpace, MemoryTraits, UseSIMD>;

  KOKKOS_FUNCTION FieldTraits() = default;

  KOKKOS_FUNCTION FieldTraits(const ViewType &view)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(view) {}

  KOKKOS_FUNCTION FieldTraits(const ViewType &view1, const ViewType &view2)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(view1, view2) {}

  KOKKOS_FUNCTION FieldTraits(const ViewType &view1, const ViewType &view2,
                              const ViewType &view3)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(view1, view2,
                                                            view3) {}

  template <typename MemberType>
  KOKKOS_FUNCTION FieldTraits(const MemberType &team)
      : ImplFieldTraits<ViewType, StoreDisplacement, StoreVelocity,
                        StoreAcceleration, StoreMassMatrix>(team) {}
};

} // namespace impl

template <int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix, bool UseSIMD = false>
struct field : impl::FieldTraits<NGLL, DimensionTag, MediumTag, MemorySpace,
                                 MemoryTraits, StoreDisplacement, StoreVelocity,
                                 StoreAcceleration, StoreMassMatrix, UseSIMD> {
  using ViewType =
      typename impl::FieldTraits<NGLL, DimensionTag, MediumTag, MemorySpace,
                                 MemoryTraits, StoreDisplacement, StoreVelocity,
                                 StoreAcceleration, StoreMassMatrix,
                                 UseSIMD>::ViewType;

  constexpr static int ngll = NGLL;
  constexpr static int components = ViewType::components;

  constexpr static bool store_displacement = StoreDisplacement;
  constexpr static bool store_velocity = StoreVelocity;
  constexpr static bool store_acceleration = StoreAcceleration;
  constexpr static bool store_mass_matrix = StoreMassMatrix;
  constexpr static auto medium_tag = MediumTag;

  constexpr static bool isPointFieldType = false;
  constexpr static bool isElementFieldType = true;
  constexpr static bool isChunkFieldType = false;

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION field(const ViewType &view)
      : impl::FieldTraits<NGLL, DimensionTag, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(view) {}

  KOKKOS_FUNCTION field(const ViewType &view1, const ViewType &view2)
      : impl::FieldTraits<NGLL, DimensionTag, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(view1,
                                                                       view2) {}

  KOKKOS_FUNCTION field(const ViewType &view1, const ViewType &view2,
                        const ViewType &view3)
      : impl::FieldTraits<NGLL, DimensionTag, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(
            view1, view2, view3) {}

  template <typename MemberType>
  KOKKOS_FUNCTION field(const MemberType &team)
      : impl::FieldTraits<NGLL, DimensionTag, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(team) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
};

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
//              MemoryTraits, true, true, true, false> {
//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = true;
//   constexpr static bool store_velocity = true;
//   constexpr static bool store_acceleration = true;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType displacement;
//   ViewType velocity;
//   ViewType acceleration;

//   // Enable only if the memory space is scratch space
//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : displacement(team.team_scratch(0)), velocity(team.team_scratch(0)),
//         acceleration(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &displacement, const ViewType
//   &velocity,
//                         const ViewType &acceleration)
//       : displacement(displacement), velocity(velocity),
//         acceleration(acceleration) {};

//   static int shmem_size() { return 3 * ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
//              MemoryTraits, true, false, false, false> {
//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = true;
//   constexpr static bool store_velocity = false;
//   constexpr static bool store_acceleration = false;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType displacement;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : displacement(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &displacement)
//       : displacement(displacement) {};

//   static int shmem_size() { return ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
//              MemoryTraits, false, true, false, false> {
//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = false;
//   constexpr static bool store_velocity = true;
//   constexpr static bool store_acceleration = false;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType velocity;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : velocity(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &velocity) : velocity(velocity) {};

//   static int shmem_size() { return ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
//              MemoryTraits, false, false, true, false> {
//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim2,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = false;
//   constexpr static bool store_velocity = false;
//   constexpr static bool store_acceleration = true;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType acceleration;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : acceleration(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &acceleration)
//       : acceleration(acceleration) {};

//   static int shmem_size() { return ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
//              MemoryTraits, true, true, false, false> {
//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim3,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = true;
//   constexpr static bool store_velocity = true;
//   constexpr static bool store_acceleration = false;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType displacement;
//   ViewType velocity;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : displacement(team.team_scratch(0)), velocity(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &displacement, const ViewType
//   &velocity)
//       : displacement(displacement), velocity(velocity) {};

//   static int shmem_size() { return 2 * ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
//              MemoryTraits, true, false, true, false> {

//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim3,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = true;
//   constexpr static bool store_velocity = false;
//   constexpr static bool store_acceleration = true;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType displacement;
//   ViewType acceleration;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : displacement(team.team_scratch(0)),
//       acceleration(team.team_scratch(0)) {
//   }

//   KOKKOS_FUNCTION field(const ViewType &displacement,
//                         const ViewType &acceleration)
//       : displacement(displacement), acceleration(acceleration) {};

//   static int shmem_size() { return 2 * ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
//              MemoryTraits, false, true, true, false> {

//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim3,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = false;
//   constexpr static bool store_velocity = true;
//   constexpr static bool store_acceleration = true;
//   constexpr static bool store_mass_matrix = false;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType velocity;
//   ViewType acceleration;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : velocity(team.team_scratch(0)), acceleration(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &velocity, const ViewType
//   &acceleration)
//       : velocity(velocity), acceleration(acceleration) {};

//   static int shmem_size() { return 2 * ViewType::shmem_size(); }
// };

// template <int NGLL, specfem::element::medium_tag MediumTag,
//           typename MemorySpace, typename MemoryTraits>
// struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
//              MemoryTraits, false, false, false, true> {
//   static constexpr int components =
//       specfem::medium::medium<specfem::dimension::type::dim3,
//                               MediumTag>::components;

//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto dimension_type = specfem::dimension::type::dim2;
//   constexpr static int ngll = NGLL;
//   constexpr static bool store_displacement = false;
//   constexpr static bool store_velocity = false;
//   constexpr static bool store_acceleration = false;
//   constexpr static bool store_mass_matrix = true;

//   using memory_space = MemorySpace;

//   using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
//                                 Kokkos::LayoutRight, MemorySpace,
//                                 MemoryTraits>;

//   ViewType mass_matrix;

//   template <typename MemberType,
//             std::enable_if_t<std::is_same<typename
//             MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>::value,
//                              int> = 0>
//   KOKKOS_FUNCTION field(const MemberType &team)
//       : mass_matrix(team.team_scratch(0)) {}

//   KOKKOS_FUNCTION field(const ViewType &mass_matrix)
//       : mass_matrix(mass_matrix) {};

//   static int shmem_size() { return ViewType::shmem_size(); }
// };
} // namespace element
} // namespace specfem

#endif // _ELEMENT_FIELD_HPP
