#ifndef _ELEMENT_FIELD_HPP
#define _ELEMENT_FIELD_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace element {
template <int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix>
struct field;

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
             MemoryTraits, true, true, true, false> {
  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

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

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
             MemoryTraits, true, false, false, false> {
  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType displacement;

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

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
             MemoryTraits, false, true, false, false> {
  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType velocity;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : velocity(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &velocity) : velocity(velocity){};

  static int shmem_size() { return ViewType::shmem_size(); }
};

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim2, MediumTag, MemorySpace,
             MemoryTraits, false, false, true, false> {
  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType acceleration;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : acceleration(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &acceleration)
      : acceleration(acceleration){};

  static int shmem_size() { return ViewType::shmem_size(); }
};

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
             MemoryTraits, true, true, false, false> {
  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim3,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType displacement;
  ViewType velocity;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : displacement(team.team_scratch(0)), velocity(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &displacement, const ViewType &velocity)
      : displacement(displacement), velocity(velocity){};

  static int shmem_size() { return 2 * ViewType::shmem_size(); }
};

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
             MemoryTraits, true, false, true, false> {

  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim3,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType displacement;
  ViewType acceleration;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : displacement(team.team_scratch(0)), acceleration(team.team_scratch(0)) {
  }

  KOKKOS_FUNCTION field(const ViewType &displacement,
                        const ViewType &acceleration)
      : displacement(displacement), acceleration(acceleration){};

  static int shmem_size() { return 2 * ViewType::shmem_size(); }
};

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
             MemoryTraits, false, true, true, false> {

  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim3,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType velocity;
  ViewType acceleration;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : velocity(team.team_scratch(0)), acceleration(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &velocity, const ViewType &acceleration)
      : velocity(velocity), acceleration(acceleration){};

  static int shmem_size() { return 2 * ViewType::shmem_size(); }
};

template <int NGLL, specfem::element::medium_tag MediumTag,
          typename MemorySpace, typename MemoryTraits>
struct field<NGLL, specfem::dimension::type::dim3, MediumTag, MemorySpace,
             MemoryTraits, false, false, false, true> {
  static constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim3,
                              MediumTag>::components;

  using ViewType = Kokkos::View<type_real[NGLL][NGLL][components],
                                Kokkos::LayoutRight, MemorySpace, MemoryTraits>;

  ViewType mass_matrix;

  template <typename MemberType,
            std::enable_if_t<std::is_same<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>::value,
                             int> = 0>
  KOKKOS_FUNCTION field(const MemberType &team)
      : mass_matrix(team.team_scratch(0)) {}

  KOKKOS_FUNCTION field(const ViewType &mass_matrix)
      : mass_matrix(mass_matrix){};

  static int shmem_size() { return ViewType::shmem_size(); }
};
} // namespace element
} // namespace specfem

#endif // _ELEMENT_FIELD_HPP
