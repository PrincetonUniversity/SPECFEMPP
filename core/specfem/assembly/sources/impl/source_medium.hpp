#pragma once

#include "source/source.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// Forward declarations
namespace specfem {
namespace algorithms {
template <specfem::dimension::type DimensionTag>
specfem::point::local_coordinates<DimensionTag> locate_point(
    const specfem::point::global_coordinates<DimensionTag> &coordinates,
    const specfem::assembly::mesh<DimensionTag> &mesh);
}
} // namespace specfem

namespace specfem::assembly::sources_impl {
/**
 * @brief Information about sources located within a medium
 *
 * @tparam Dimension Dimension of spectral elements
 * @tparam Medium Medium type
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
struct source_medium {

private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using SourceTimeFunctionView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store source time
                                                   ///< functions

  using SourceArrayView =
      Kokkos::View<type_real ****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store source arrays

  constexpr static int components =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::components; ///< Number
                                                           ///< of
                                                           ///< components
                                                           ///< in the
                                                           ///< medium

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto medium_tag = MediumTag; ///< Medium type
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of spectral elements
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
  source_medium() = default;

  /**
   * @brief Compute source information for a given medium
   *
   * @param sources Vector of sources located within the medium
   * @param mesh Finite element mesh information
   * @param jacobian_matrix Jacobian matrix for every quadrature point
   * @param element_types Element types for every spectral element
   * @param t0 Initial time
   * @param dt Time step
   * @param nsteps Number of time steps
   */
  source_medium(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::assembly::jacobian_matrix &jacobian_matrix,
      const specfem::assembly::element_types &element_types, const type_real t0,
      const type_real dt, const int nsteps);
  ///@}

  IndexView source_index_mapping; ///< Spectral element index for every source
  IndexView::HostMirror h_source_index_mapping; ///< Host mirror of
                                                ///< source_index_mapping
  SourceTimeFunctionView source_time_function;  ///< Source time function for
                                                ///< every source
  SourceTimeFunctionView::HostMirror
      h_source_time_function;   ///< Host mirror of source_time_function
  SourceArrayView source_array; ///< Lagrange interpolants for every source
  SourceArrayView::HostMirror h_source_array; ///< Host mirror of source_array

  template <typename IndexType, typename PointSourceType>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const {
    /* For the source it is important to remember that we are using the
     * mapped index to access the element and source indices
     * that means that index actually is a mapped_chunk_index
     * and we need to use index.ispec to access the element index
     * and index.imap to access the source index
     */
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          source_array(isource, component, index.iz, index.ix);
    }
  }

  template <typename IndexType, typename PointSourceType>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const {
    /* For the source it is important to remember that we are using the
     * mapped index to access the element and source indices
     * that means that index actually is a mapped_chunk_index
     * and we need to use index.ispec to access the element index
     * and index.imap to access the source index
     */
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      source_time_function(timestep, isource, component) =
          point_source.stf(component);
      source_array(isource, component, index.iz, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }

  template <typename IndexType, typename PointSourceType>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          h_source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          h_source_array(isource, component, index.iz, index.ix);
    }
  }

  template <typename IndexType, typename PointSourceType>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      h_source_time_function(timestep, isource, component) =
          point_source.stf(component);
      h_source_array(isource, component, index.iz, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }
};
} // namespace specfem::assembly::sources_impl
