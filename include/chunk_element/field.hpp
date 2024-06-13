#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace chunk_element {

namespace impl {
template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix>
struct FieldTraits {
public:
  constexpr static int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumTag>::components;
  constexpr static int num_elements = NumElements;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto dimension_type = DimensionType;
  constexpr static bool store_displacement = StoreDisplacement;
  constexpr static bool store_velocity = StoreVelocity;
  constexpr static bool store_acceleration = StoreAcceleration;
  constexpr static bool store_mass_matrix = StoreMassMatrix;
  constexpr static int ngll = NGLL;

  using ViewType = Kokkos::View<type_real[NumElements][NGLL][NGLL][components],
                                Kokkos::LayoutLeft, MemorySpace, MemoryTraits>;
};
} // namespace impl

template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix>
struct field;

template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits>
struct field<NumElements, NGLL, DimensionType, MediumTag, MemorySpace,
             MemoryTraits, true, true, true, false>
    : public impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                               MemorySpace, MemoryTraits, true, true, true,
                               false> {

  using memory_space = MemorySpace;

  using ViewType =
      typename impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                                 MemorySpace, MemoryTraits, true, true, true,
                                 false>::ViewType;

  ViewType displacement;
  ViewType velocity;
  ViewType acceleration;

  // Enable only if the memory space is scratch space
  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : displacement(team.team_scratch(0)), velocity(team.team_scratch(0)),
        acceleration(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &displacement, const ViewType &velocity,
                        const ViewType &acceleration)
      : displacement(displacement), velocity(velocity),
        acceleration(acceleration){};

  static int shmem_size() { return 3 * ViewType::shmem_size(); }
};

template <int NumElements, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits>
struct field<NumElements, NGLL, DimensionType, MediumTag, MemorySpace,
             MemoryTraits, true, false, false, false>
    : public impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                               MemorySpace, MemoryTraits, true, false, false,
                               false> {

  using memory_space = MemorySpace;

  using ViewType =
      typename impl::FieldTraits<NumElements, NGLL, DimensionType, MediumTag,
                                 MemorySpace, MemoryTraits, true, false, false,
                                 false>::ViewType;

  ViewType displacement;

  // Enable only if the memory space is scratch space
  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : displacement(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &displacement)
      : displacement(displacement){};

  static int shmem_size() { return ViewType::shmem_size(); }
};

} // namespace chunk_element
} // namespace specfem
