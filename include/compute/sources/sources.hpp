#ifndef _COMPUTE_SOURCES_2_HPP
#define _COMPUTE_SOURCES_2_HPP

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
#include "source/source.hpp"
#include "source_medium.hpp"

namespace specfem {
namespace compute {
struct sources {

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  sources() = default;

  /**
   * @brief Generate source information for a given mesh
   *
   * @param sources Vector of sources read from sources file
   * @param mesh Finite element mesh information
   * @param partial_derivatives Partial derivatives for every quadrature point
   * @param properties Material properties for every quadrature point
   * @param t0 Initial time
   * @param dt Time step
   * @param nsteps Number of time steps
   */
  sources(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties, const type_real t0,
      const type_real dt, const int nsteps);
  ///@}

  /**
   * @brief Get the information on sources for a given medium
   *
   * @tparam Medium Medium type
   * @return specfem::compute::impl::sources::source_medium<
   * specfem::dimension::type::dim2, Medium>  Source information for the medium
   */
  template <specfem::element::medium_tag Medium>
  inline specfem::compute::source_medium<specfem::dimension::type::dim2, Medium>
  get_source_medium() const {
    if constexpr (Medium == specfem::element::medium_tag::acoustic) {
      return acoustic_sources;
    } else if constexpr (Medium == specfem::element::medium_tag::elastic) {
      return elastic_sources;
    } else {
      static_assert("Invalid medium type");
    }
  }

  int nsources; ///< Number of sources
  specfem::kokkos::HostView1d<int>
      source_domain_index_mapping; ///< Spectral element index for every source
  specfem::kokkos::HostView1d<specfem::element::medium_tag>
      source_medium_mapping; ///< Medium type for every spectral element where
                             ///< source is located
  specfem::kokkos::HostView1d<specfem::wavefield::type>
      source_wavefield_mapping; ///< Wavefield type on which any source acts
  specfem::compute::source_medium<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::acoustic>
      acoustic_sources; ///< Information for sources within acoustic medium
  specfem::compute::source_medium<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic>
      elastic_sources; ///< Information for sources within elastic medium
};
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_SOURCES_2_HPP */
