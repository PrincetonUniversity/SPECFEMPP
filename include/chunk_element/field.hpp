#pragma once

#include "datatypes/chunk_element_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace chunk_element {

namespace impl {

template <typename ViewType, bool StoreDisplacement> struct Displacement;

template <typename ViewType> struct Displacement<ViewType, true> {
  ViewType displacement;

  KOKKOS_FUNCTION Displacement() = default;

  KOKKOS_FUNCTION Displacement(const ViewType &displacement)
      : displacement(displacement) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Displacement(const ScratchMemorySpace &scratch_space)
      : displacement(scratch_space) {}
};

template <typename ViewType> struct Displacement<ViewType, false> {};

template <typename ViewType, bool StoreVelocity> struct Velocity;

template <typename ViewType> struct Velocity<ViewType, true> {
  ViewType velocity;

  KOKKOS_FUNCTION Velocity() = default;

  KOKKOS_FUNCTION Velocity(const ViewType &velocity) : velocity(velocity) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Velocity(const ScratchMemorySpace &scratch_space)
      : velocity(scratch_space) {}
};

template <typename ViewType> struct Velocity<ViewType, false> {};

template <typename ViewType, bool StoreAcceleration> struct Acceleration;

template <typename ViewType> struct Acceleration<ViewType, true> {
  ViewType acceleration;

  KOKKOS_FUNCTION Acceleration() = default;

  KOKKOS_FUNCTION Acceleration(const ViewType &acceleration)
      : acceleration(acceleration) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Acceleration(const ScratchMemorySpace &scratch_space)
      : acceleration(scratch_space) {}
};

template <typename ViewType> struct Acceleration<ViewType, false> {};

template <typename ViewType, bool StoreMassMatrix> struct MassMatrix;

template <typename ViewType> struct MassMatrix<ViewType, true> {
  ViewType mass_matrix;

  KOKKOS_FUNCTION MassMatrix() = default;

  KOKKOS_FUNCTION MassMatrix(const ViewType &mass_matrix)
      : mass_matrix(mass_matrix) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION MassMatrix(const ScratchMemorySpace &scratch_space)
      : mass_matrix(scratch_space) {}
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

template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix, bool UseSIMD>
struct FieldTraits
    : public ImplFieldTraits<
          specfem::datatype::ScalarChunkViewType<
              type_real, NumElements, NGLL,
              specfem::medium::medium<DimensionType, MediumTag>::components,
              MemorySpace, MemoryTraits, UseSIMD>,
          StoreDisplacement, StoreVelocity, StoreAcceleration,
          StoreMassMatrix> {

  constexpr static int components =
      specfem::medium::medium<DimensionType, MediumTag>::components;

  using ViewType =
      specfem::datatype::ScalarChunkViewType<type_real, NumElements, NGLL,
                                             components, MemorySpace,
                                             MemoryTraits, UseSIMD>;

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

template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix, bool UseSIMD>
struct field
    : public impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                               MemorySpace, MemoryTraits, StoreDisplacement,
                               StoreVelocity, StoreAcceleration,
                               StoreMassMatrix, UseSIMD> {

public:
  using ViewType =
      typename impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                                 MemorySpace, MemoryTraits, StoreDisplacement,
                                 StoreVelocity, StoreAcceleration,
                                 StoreMassMatrix, UseSIMD>::ViewType;

  constexpr static int components = ViewType::components;
  constexpr static auto medium_tag = MediumTag;
  constexpr static int dimension =
      specfem::dimension::dimension<DimensionType>::dim;
  constexpr static int num_elements = NumElements;
  constexpr static int ngll = NGLL;

  constexpr static bool store_displacement = StoreDisplacement;
  constexpr static bool store_velocity = StoreVelocity;
  constexpr static bool store_acceleration = StoreAcceleration;
  constexpr static bool store_mass_matrix = StoreMassMatrix;

  constexpr static bool isChunkFieldType = true;
  constexpr static bool isPointFieldType = false;
  constexpr static bool isElementFieldType = false;

  using memory_space = MemorySpace;
  using simd = specfem::datatype::simd<type_real, UseSIMD>;

  KOKKOS_FUNCTION field(const ViewType &view)
      : impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                          MemorySpace, MemoryTraits, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(view) {}

  KOKKOS_FUNCTION field(const ViewType &view1, const ViewType &view2)
      : impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                          MemorySpace, MemoryTraits, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(view1, view2) {}

  KOKKOS_FUNCTION field(const ViewType &view1, const ViewType &view2,
                        const ViewType &view3)
      : impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                          MemorySpace, MemoryTraits, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(view1, view2, view3) {}

  template <typename MemberType>
  KOKKOS_FUNCTION field(const MemberType &team)
      : impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                          MemorySpace, MemoryTraits, StoreDisplacement,
                          StoreVelocity, StoreAcceleration, StoreMassMatrix,
                          UseSIMD>(team) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
};

// template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag, typename MemorySpace,
//           typename MemoryTraits>
// struct field<NumElements, NGLL, DimensionType, MediumTag, MemorySpace,
//              MemoryTraits, true, true, true, false>
//     : public impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
//                                MemorySpace, MemoryTraits, true, true, true,
//                                false> {

//   using memory_space = MemorySpace;

//   using ViewType =
//       typename impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
//                                  MemorySpace, MemoryTraits, true, true, true,
//                                  false>::ViewType;

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

// template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag, typename MemorySpace,
//           typename MemoryTraits>
// struct field<NumElements, NGLL, DimensionType, MediumTag, MemorySpace,
//              MemoryTraits, true, false, false, false>
//     : public impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
//                                MemorySpace, MemoryTraits, true, false, false,
//                                false> {

//   using memory_space = MemorySpace;

//   using ViewType =
//       typename impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
//                                  MemorySpace, MemoryTraits, true, false,
//                                  false, false>::ViewType::type;

//   ViewType displacement;

//   // Enable only if the memory space is scratch space
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

} // namespace chunk_element
} // namespace specfem
