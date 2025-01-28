#pragma once

#include "datatypes/chunk_edge_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace chunk_edge {

namespace impl {

template <typename ViewType, bool StoreDisplacement> struct Displacement {
  ViewType displacement; ///< Displacement for every quadrature point within the
                         ///< chunk. Defined if StoreDisplacement is true.

  KOKKOS_FUNCTION Displacement() = default;

  KOKKOS_FUNCTION Displacement(const ViewType &displacement)
      : displacement(displacement) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Displacement(const ScratchMemorySpace &scratch_space)
      : displacement(scratch_space) {}
};

template <typename ViewType> struct Displacement<ViewType, false> {};

template <typename ViewType, bool StoreVelocity> struct Velocity {
  ViewType velocity; ///< Velocity for every quadrature point within the chunk.
                     ///< Defined if StoreVelocity is true.

  KOKKOS_FUNCTION Velocity() = default;

  KOKKOS_FUNCTION Velocity(const ViewType &velocity) : velocity(velocity) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Velocity(const ScratchMemorySpace &scratch_space)
      : velocity(scratch_space) {}
};

template <typename ViewType> struct Velocity<ViewType, false> {};

template <typename ViewType, bool StoreAcceleration> struct Acceleration {
  ViewType acceleration; ///< Acceleration for every quadrature point within the
                         ///< chunk. Defined if StoreAcceleration is true.

  KOKKOS_FUNCTION Acceleration() = default;

  KOKKOS_FUNCTION Acceleration(const ViewType &acceleration)
      : acceleration(acceleration) {}

  template <typename ScratchMemorySpace>
  KOKKOS_FUNCTION Acceleration(const ScratchMemorySpace &scratch_space)
      : acceleration(scratch_space) {}
};

template <typename ViewType> struct Acceleration<ViewType, false> {};

template <typename ViewType, bool StoreMassMatrix> struct MassMatrix {
  ViewType mass_matrix; ///< Mass matrix for every quadrature point within the
                        ///< chunk. Defined if StoreMassMatrix is true.

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

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int  Amount of shared memory required
   */
  constexpr static int shmem_size() {
    return (static_cast<int>(StoreDisplacement) +
            static_cast<int>(StoreVelocity) +
            static_cast<int>(StoreAcceleration) +
            static_cast<int>(StoreMassMatrix)) *
           ViewType::shmem_size();
  }
};

template <int NumEdges, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix, bool UseSIMD>
struct FieldTraits
    : public ImplFieldTraits<specfem::datatype::ScalarChunkEdgeViewType<
                                 type_real, NumEdges, NGLL,
                                 specfem::element::attributes<
                                     DimensionType, MediumTag>::components(),
                                 MemorySpace, MemoryTraits, UseSIMD>,
                             StoreDisplacement, StoreVelocity,
                             StoreAcceleration, StoreMassMatrix> {

  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();

  using ViewType =
      specfem::datatype::ScalarChunkEdgeViewType<type_real, NumEdges, NGLL,
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

/**
 * @brief Chunk Edge class for storing displacement, velocity, acceleration,
 * and mass matrix within a chunk of edges.
 *
 * @tparam NumEdges Number of edges in the chunk
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points
 * @tparam DimensionType Dimension of the medium
 * @tparam MediumTag Medium tag for the edges within the chunk
 * @tparam MemorySpace Memory space for the views
 * @tparam MemoryTraits Memory traits for the views
 * @tparam StoreDisplacement Boolean to indicate if displacement should be
 * stored
 * @tparam StoreVelocity Boolean to indicate if velocity should be stored
 * @tparam StoreAcceleration Boolean to indicate if acceleration should be
 * stored
 * @tparam StoreMassMatrix Boolean to indicate if mass matrix should be stored
 * @tparam UseSIMD Boolean to indicate to use SIMD instructions
 */
template <int NumEdges, int NGLL, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename MemorySpace,
          typename MemoryTraits, bool StoreDisplacement, bool StoreVelocity,
          bool StoreAcceleration, bool StoreMassMatrix, bool UseSIMD>
struct field
    : public impl::FieldTraits<NumEdges, NGLL, DimensionType, MediumTag,
                               MemorySpace, MemoryTraits, StoreDisplacement,
                               StoreVelocity, StoreAcceleration,
                               StoreMassMatrix, UseSIMD> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{

  /**
   * @brief Underlying View type used to store the field data.
   */
  using ViewType =
      typename impl::FieldTraits<NumEdges, NGLL, DimensionType, MediumTag,
                                 MemorySpace, MemoryTraits, StoreDisplacement,
                                 StoreVelocity, StoreAcceleration,
                                 StoreMassMatrix, UseSIMD>::ViewType;
  using memory_space = MemorySpace; ///< Memory space for the views
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

  ///@}

  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static int components = ViewType::components; ///< Number of
                                                          ///< components
  constexpr static auto medium_tag = MediumTag;    ///< Medium tag for the
                                                   ///< edges within the chunk
  constexpr static auto dimension = DimensionType; ///< Dimension of the
                                                   ///< edges within the chunk
  constexpr static int num_edges = NumEdges;       ///< Number of edges in
                                                   ///< the chunk
  constexpr static int ngll = NGLL; ///< Number of Gauss-Lobatto-Legendre
                                    ///< points

  constexpr static bool store_displacement =
      StoreDisplacement; ///< Boolean to indicate if displacement should be
                         ///< stored
  constexpr static bool store_velocity = StoreVelocity; ///< Boolean to indicate
                                                        ///< if velocity should
                                                        ///< be stored
  constexpr static bool store_acceleration =
      StoreAcceleration;                                     ///< Boolean to
                                                             ///< indicate if
                                                             ///< acceleration
                                                             ///< should be
                                                             ///< stored
  constexpr static bool store_mass_matrix = StoreMassMatrix; ///< Boolean to
                                                             ///< indicate if
                                                             ///< mass matrix
                                                             ///< should be
                                                             ///< stored

  constexpr static bool isChunkFieldType = false; ///< Boolean to indicate if
                                                  ///< this is a chunk field
  constexpr static bool isPointFieldType = false; ///< Boolean to indicate if
                                                  ///< this is a point field
  constexpr static bool isElementFieldType =
      false; ///< Boolean to indicate if
             ///< this is an element field
  constexpr static bool isChunkEdgeFieldType = true;
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Constructor a new chunk field with a single view.
   *
   * Enabled when only one of StoreDisplacement, StoreVelocity,
   * StoreAcceleration, or StoreMassMatrix is true.
   *
   * @param view View to initialize the field with.
   */
  KOKKOS_FUNCTION field(const ViewType &view)
      : impl::FieldTraits<NumEdges, NGLL, DimensionType, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(view) {}

  /**
   * @brief Constructor a new chunk field with two views.
   *
   * Enabled when only two of StoreDisplacement, StoreVelocity, or
   * StoreAcceleration are true.
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
  KOKKOS_FUNCTION field(const ViewType &view1, const ViewType &view2)
      : impl::FieldTraits<NumEdges, NGLL, DimensionType, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(view1,
                                                                       view2) {}

  /**
   * @brief Constructor a new chunk field with three views.
   *
   * Enabled when StoreDisplacement, StoreVelocity, and StoreAcceleration are
   * true.
   *
   * @param view1 Displacement view
   * @param view2  Velocity view
   * @param view3  Acceleration view
   */
  KOKKOS_FUNCTION field(const ViewType &view1, const ViewType &view2,
                        const ViewType &view3)
      : impl::FieldTraits<NumEdges, NGLL, DimensionType, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(
            view1, view2, view3) {}

  /**
   * @brief Construct a new chunk field object using within a Scratch Memory
   * Space
   *
   * @tparam MemberType Kokkos Team Member Type
   * @param team Kokkos Team Member where the field will be allocated
   */
  template <typename MemberType>
  KOKKOS_FUNCTION field(const MemberType &team)
      : impl::FieldTraits<NumEdges, NGLL, DimensionType, MediumTag, MemorySpace,
                          MemoryTraits, StoreDisplacement, StoreVelocity,
                          StoreAcceleration, StoreMassMatrix, UseSIMD>(team) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
  ///@}
};

} // namespace chunk_edge
} // namespace specfem
