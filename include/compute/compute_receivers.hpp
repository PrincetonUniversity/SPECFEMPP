#ifndef _COMPUTE_RECEIVERS_HPP
#define _COMPUTE_RECEIVERS_HPP

#include "kokkos_abstractions.h"
#include "quadrature.h"
#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief This struct is used to store receiver arrays required to interpolate
 * fields during seismogram calculations
 *
 */
struct receivers {
  specfem::kokkos::DeviceView4d<type_real> receiver_array;   ///< Array to store
                                                             ///< lagrange
                                                             ///< interpolants
                                                             ///< for receivers
                                                             ///< on the device
  specfem::kokkos::HostMirror4d<type_real> h_receiver_array; ///< Array to
                                                             ///< store
                                                             ///< lagrange
                                                             ///< interpolants
                                                             ///< for sources
                                                             ///< stored on
                                                             ///< host
  specfem::kokkos::DeviceView1d<int> ispec_array;   ///< Spectral element number
                                                    ///< where the source lies
                                                    ///< stored on device
  specfem::kokkos::HostMirror1d<int> h_ispec_array; ///< Spectral element
                                                    ///< number where the
                                                    ///< source lies stored on
                                                    ///< host
  specfem::kokkos::DeviceView1d<type_real> cos_recs;   ///< consine of angle
                                                       ///< used to rotate
                                                       ///< receiver components
                                                       ///< stored on device
  specfem::kokkos::HostMirror1d<type_real> h_cos_recs; ///< consine of angle
                                                       ///< used to rotate
                                                       ///< receiver
                                                       ///< components stored
                                                       ///< on host
  specfem::kokkos::DeviceView1d<type_real> sin_recs;   ///< sine of angle used
                                                       ///< to rotate receiver
                                                       ///< components stored on
                                                       ///< device
  specfem::kokkos::HostMirror1d<type_real> h_sin_recs; ///< sine of angle used
                                                       ///< to rotate receiver
                                                       ///< components stored
                                                       ///< on host
  specfem::kokkos::DeviceView5d<type_real> field;      ///< Container to store
                                                  ///< spectral element field
                                                  ///< used in computing
                                                  ///< seismograms stored on
                                                  ///< the device
  specfem::kokkos::DeviceView4d<type_real> seismogram;   ///< Container to store
                                                         ///< computed
                                                         ///< seismograms stored
                                                         ///< on the device
  specfem::kokkos::HostMirror4d<type_real> h_seismogram; ///< Container to
                                                         ///< store computed
                                                         ///< seismograms
                                                         ///< stored on the
                                                         ///< device
  specfem::kokkos::DeviceView1d<specfem::seismogram::type>
      seismogram_types; ///< Types of seismograms to be calculated stored on
                        ///< the device
  specfem::kokkos::HostMirror1d<specfem::seismogram::type>
      h_seismogram_types; ///< Types of seismograms to be calculated stored on
                          ///< the host

  /**
   * @brief Default constructor
   *
   */
  receivers(){};
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param receivers Pointer to receivers objects read from sources file
   * @param stypes Types of seismograms to be written
   * @param quadx Quarature object in x dimension
   * @param quadz Quadrature object in z dimension
   * @param mpi Pointer to the MPI object
   */
  receivers(const std::vector<specfem::receivers::receiver *> &receivers,
            const std::vector<specfem::seismogram::type> &stypes,
            const specfem::quadrature::quadrature &quadx,
            const specfem::quadrature::quadrature &quadz, const type_real xmax,
            const type_real xmin, const type_real zmax, const type_real zmin,
            const int max_sig_step, specfem::MPI::MPI *mpi);
  /**
   * @brief Sync views within this struct from host to device
   *
   * Sync views after they have been initialized on the host
   *
   */
  void sync_views();

  /**
   * @brief Sync calculated seismogram from device to the host
   *
   */
  void sync_seismograms();
};
} // namespace compute
} // namespace specfem

#endif
