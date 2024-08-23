#pragma once

#include "compute/compute_mesh.hpp"
#include "source/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
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
      specfem::medium::medium<Dimension, Medium>::components; ///< Number of
                                                              ///< components in
                                                              ///< the medium

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
   * @param properties Material properties for every quadrature point
   * @param t0 Initial time
   * @param dt Time step
   * @param nsteps Number of time steps
   */
  source_medium(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties, const type_real t0,
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
};
} // namespace compute
} // namespace specfem
