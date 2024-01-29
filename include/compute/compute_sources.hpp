#ifndef _COMPUTE_SOURCES_HPP
#define _COMPUTE_SOURCES_HPP

#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem {
namespace compute {
/**
 * @brief This struct is used to store arrays required to impose source
 * interaction during the time loop
 *
 * @note Does not implement moving sources yet
 *
 */
struct sources {
  specfem::kokkos::DeviceView4d<type_real> source_array; ///< Array to store
                                                         ///< lagrange
                                                         ///< interpolants for
                                                         ///< sources stored on
                                                         ///< device
  specfem::kokkos::HostMirror4d<type_real> h_source_array; ///< Array to store
                                                           ///< lagrange
                                                           ///< interpolants for
                                                           ///< sources stored
                                                           ///< on host
  specfem::kokkos::DeviceView2d<type_real> stf_array;   ///< Pointer to source
                                                        ///< time function for
                                                        ///< every source stored
                                                        ///< on device
  specfem::kokkos::HostMirror2d<type_real> h_stf_array; ///< Pointer to source
                                                        ///< time function for
                                                        ///< every source stored
                                                        ///< on host
  specfem::kokkos::DeviceView1d<int> ispec_array;   ///< Spectral element number
                                                    ///< where the source lies
                                                    ///< stored on device
  specfem::kokkos::HostMirror1d<int> h_ispec_array; ///< Spectral element number
                                                    ///< where the source lies
                                                    ///< stored on host
  /**
   * @brief Default constructor
   *
   */
  sources(){};
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param sources Pointer to sources objects read from sources file
   * @param quadx Quarature object in x dimension
   * @param quadz Quadrature object in z dimension
   * @param mpi Pointer to the MPI object
   */
  sources(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties, const int nsteps);
  // /**
  //  * @brief Helper routine to sync views within this struct
  //  *
  //  */
  // void sync_views();
};

} // namespace compute
} // namespace specfem

#endif
