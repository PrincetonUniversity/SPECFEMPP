#pragma once

#include "sources/impl/source_medium.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/source.hpp"
#include "specfem/tag_dispatch/element_combinations.hpp"

#include <Kokkos_Core.hpp>
#include <memory>
#include <tuple>
#include <vector>

namespace specfem::assembly {

/**
 * @brief Define the tag sets for source combinations in each dimension
 *
 * Specifies which medium, property, boundary, and wavefield
 * combinations are valid for sources in 2D and 3D simulations.
 */
namespace sources_impl {

/**
 * @brief Source tag set definitions for dimension-specific simulations
 * @tparam DimensionTag Spatial dimension
 */
template <specfem::element::dimension_tag DimensionTag> struct SourceSets;

/**
 * @brief 2D source tag sets
 *
 * Defines valid combinations of medium, property, boundary, and wavefield
 * types for 2D simulations.
 */
template <> struct SourceSets<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr static auto medium_set =
      MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic, elastic_psv_t);
  constexpr static auto property_set =
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat);
  constexpr static auto boundary_set = BOUNDARY_SET(
      none, acoustic_free_surface, stacey, composite_stacey_dirichlet);
  constexpr static auto wavefield_set =
      WAVEFIELD_SET(forward, backward, adjoint);
};

/**
 * @brief 3D source tag sets
 *
 * Defines valid combinations of medium, property, boundary, and wavefield
 * types for 3D simulations.
 */
template <> struct SourceSets<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto medium_set = MEDIUM_SET(elastic, acoustic);
  constexpr static auto property_set = PROPERTY_SET(isotropic);
  constexpr static auto boundary_set = BOUNDARY_SET(none);
  constexpr static auto wavefield_set =
      WAVEFIELD_SET(forward, backward, adjoint);
};
} // namespace sources_impl

/**
 * @brief Assembly-level source management for spectral element simulations
 *
 * This template class manages sources within assembled finite element meshes,
 * providing efficient access to source data for both host and device
 * computations. The sources are organized by medium type (elastic, acoustic,
 * poroelastic) and support time-dependent source time functions.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3)
 */
// clang-format off
template <specfem::element::dimension_tag DimensionTag>
struct sources {

public:
  /**
   * @name Public Constants
   */
  ///@{
  /**
   * @brief Dimension tag for this source assembly
   */
  constexpr static auto dimension_tag = DimensionTag;
  ///@}

private:
  /**
   * @name Private Type Definitions
   *
   * Kokkos view types used for efficient memory management and data access
   * patterns in source computations.
   */
  ///@{

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using MediumTagViewType = Kokkos::View<specfem::element::medium_tag *,
                                         Kokkos::DefaultExecutionSpace>;
  using WavefieldTagViewType = Kokkos::View<specfem::simulation::field_type *,
                                            Kokkos::DefaultExecutionSpace>;
  using BoundaryTagViewType = Kokkos::View<specfem::element::boundary_tag *,
                                           Kokkos::DefaultExecutionSpace>;
  using PropertyTagViewType = Kokkos::View<specfem::element::property_tag *,
                                           Kokkos::DefaultExecutionSpace>;

  ///@}

public:
  /**
   * @name Constructors and Destructors
   */
  ///@{

  /**
   * @brief Default constructor
   */
  sources() = default;

  /**
   * @brief Construct source assembly from mesh and source configuration
   *
   * @param sources Vector of source objects
   * @param mesh Finite element mesh
   * @param jacobian_matrix Jacobian transformation matrices
   * @param element_types Element classification by medium and property
   * @param t0 Initial simulation time
   * @param dt Time step size
   * @param nsteps Total number of time steps
   */
  sources(
      std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
          &sources,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
      const specfem::assembly::element_types<DimensionTag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);
  ///@}

  /**
   * @brief Retrieve source indices for specified criteria on host memory
   *
   * @param medium Physical medium type
   * @param property Material property classification
   * @param boundary Boundary condition type
   * @param wavefield Simulation type (forward, adjoint, backward)
   * @return Tuple of (element indices, source indices) host views
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
  get_sources_on_host(const specfem::element::medium_tag medium,
                      const specfem::element::property_tag property,
                      const specfem::element::boundary_tag boundary,
                      const specfem::simulation::field_type wavefield) const;

  /**
   * @brief Retrieve source indices for specified criteria on device memory
   *
   * @param medium Physical medium type
   * @param property Material property classification
   * @param boundary Boundary condition type
   * @param wavefield Simulation type (forward, adjoint, backward)
   * @return Tuple of (element indices, source indices) device views
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
  get_sources_on_device(const specfem::element::medium_tag medium,
                        const specfem::element::property_tag property,
                        const specfem::element::boundary_tag boundary,
                        const specfem::simulation::field_type wavefield) const;

  /**
   * @brief Get the MPI slice index for each source
   * @return Vector of MPI slice indices corresponding to each source (host memory)
   */
  const std::vector<int> &get_islice() const { return source_islice_; }

  /**
   * @brief Update the current simulation time step
   *
   * Must be called before each time step to ensure source time functions are
   * evaluated at the correct temporal point.
   *
   * @param timestep Current time step index (0-based)
   */
  void update_timestep(const int timestep) { this->timestep = timestep; }

private:
  /**
   * @name Private Data Members
   */
  ///@{

  int nspec; ///< Number of spectral elements in mesh

  IndexViewType source_domain_index_mapping; ///< Source-to-domain mapping (device)
  IndexViewType::host_mirror_type h_source_domain_index_mapping; ///< Source-to-domain mapping (host)

  IndexViewType element_indices; ///< Element index for each source (device)
  IndexViewType::host_mirror_type h_element_indices; ///< Element index for each source (host)

  IndexViewType source_indices; ///< Source indices (device)
  IndexViewType::host_mirror_type h_source_indices; ///< Source indices (host)

  MediumTagViewType medium_types; ///< Medium tag for each source (device)
  MediumTagViewType::host_mirror_type h_medium_types; ///< Medium tag for each source (host)

  WavefieldTagViewType wavefield_types; ///< Wavefield type for each source (device)
  WavefieldTagViewType::host_mirror_type h_wavefield_types; ///< Wavefield type for each source (host)

  BoundaryTagViewType boundary_types; ///< Boundary tag for each source (device)
  BoundaryTagViewType::host_mirror_type h_boundary_types; ///< Boundary tag for each source (host)

  PropertyTagViewType property_types; ///< Property tag for each source (device)
  PropertyTagViewType::host_mirror_type h_property_types; ///< Property tag for each source (host)

  ///@}

  /**
   * @name Medium-Specific Storage
   */
  ///@{

  /// Tag combinations for medium-specific source data
  static constexpr auto combinations_by_medium =
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
      sources_impl::SourceSets<DimensionTag>::medium_set;

  /// Template alias for medium-specific source data structures
  template <typename TagsType>
  using SourceMediumTemplateType =
      specfem::assembly::sources_impl::source_medium<TagsType::dimension_tag,
                                                     TagsType::medium_tag>;

  /// Storage container for sources organized by medium type
  specfem::tag_dispatch::TypedStorage<SourceMediumTemplateType,
                                      decltype(combinations_by_medium)>
      source_by_medium;

  int timestep; ///< Current simulation timestep
  std::vector<int> source_islice_; ///< MPI slice index for each source (host)

  ///@}

  /**
   * @name Index Storage by Tag Combination
   *
   * Keyed by (dimension, medium, property, boundary, wavefield)
   */
  ///@{

  /// Tag combinations for accessing sources by all attributes
  static constexpr auto combinations_by_source =
      combinations_by_medium *
      sources_impl::SourceSets<DimensionTag>::property_set *
      sources_impl::SourceSets<DimensionTag>::boundary_set *
      sources_impl::SourceSets<DimensionTag>::wavefield_set;

  using HostIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>;

  /// Element indices organized by tag combination (device)
  specfem::tag_dispatch::Storage<IndexViewType,
                                 decltype(combinations_by_source)>
      source_element_by_combination;

  /// Element indices organized by tag combination (host)
  specfem::tag_dispatch::Storage<HostIndexViewType,
                                 decltype(combinations_by_source)>
      h_source_element_by_combination;

  /// Source indices organized by tag combination (device)
  specfem::tag_dispatch::Storage<IndexViewType,
                                 decltype(combinations_by_source)>
      source_source_by_combination;

  /// Source indices organized by tag combination (host)
  specfem::tag_dispatch::Storage<HostIndexViewType,
                                 decltype(combinations_by_source)>
      h_source_source_by_combination;

  ///@}

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag D>
  friend KOKKOS_INLINE_FUNCTION void load_on_device(
      const IndexType index,
      const specfem::assembly::sources<D> &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag D>
  friend void load_on_host(
      const IndexType index,
      const specfem::assembly::sources<D> &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag D>
  friend KOKKOS_INLINE_FUNCTION void store_on_device(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<D> &sources);

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag D>
  friend void store_on_host(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<D> &sources);
};
// clang-format on

} // namespace specfem::assembly

// Include free-function templates (load/store on device/host)
#include "sources.tpp"
