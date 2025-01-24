#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/element_types/element_types.hpp"
#include "source/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
/**
 * @brief Information about sources located within a medium
 *
 * @tparam Dimension Dimension of spectral elements
 * @tparam Medium Medium type
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag Medium>
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
      specfem::element::attributes<Dimension,
                                   Medium>::components(); ///< Number
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
  constexpr static auto medium_tag = Medium; ///< Medium type
  constexpr static auto dimension =
      Dimension; ///< Dimension of spectral elements
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
   * @param partial_derivatives Partial derivatives for every quadrature point
   * @param element_types Element types for every spectral element
   * @param t0 Initial time
   * @param dt Time step
   * @param nsteps Number of time steps
   */
  source_medium(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types, const type_real t0,
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

  template <typename IteratorIndexType, typename PointSourceType>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IteratorIndexType &iterator_index,
                 PointSourceType &point_source) const {
    /* For the source it is important to remember that we are using the
     * mapped index to access the element and source indices
     * that means that index actually is a mapped_chunk_index
     * and we need to use index.ispec to access the element index
     * and index.imap to access the source index
     */
    const auto index = iterator_index.index;
    const auto isource = iterator_index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          source_array(isource, component, index.iz, index.ix);
    }
  }

  template <typename IteratorIndexType, typename PointSourceType>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IteratorIndexType iterator_index,
                  const PointSourceType &point_source) const {
    /* For the source it is important to remember that we are using the
     * mapped index to access the element and source indices
     * that means that index actually is a mapped_chunk_index
     * and we need to use index.ispec to access the element index
     * and index.imap to access the source index
     */
    const auto index = iterator_index.index;
    const auto isource = iterator_index.imap;
    for (int component = 0; component < components; component++) {
      source_time_function(timestep, isource, component) =
          point_source.stf(component);
      source_array(isource, component, index.iz, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }

  template <typename IteratorIndexType, typename PointSourceType>
  void load_on_host(const int timestep, const IteratorIndexType iterator_index,
                    PointSourceType &point_source) const {
    const auto index = iterator_index.index;
    const auto isource = iterator_index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          h_source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          h_source_array(isource, component, index.iz, index.ix);
    }
  }

  template <typename IteratorIndexType, typename PointSourceType>
  void store_on_host(const int timestep, const IteratorIndexType iterator_index,
                     const PointSourceType &point_source) const {
    const auto index = iterator_index.index;
    const auto isource = iterator_index.imap;
    for (int component = 0; component < components; component++) {
      h_source_time_function(timestep, isource, component) =
          point_source.stf(component);
      h_source_array(isource, component, index.iz, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }
};
} // namespace impl
} // namespace compute
} // namespace specfem
