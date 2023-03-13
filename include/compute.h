#ifndef COMPUTE_H
#define COMPUTE_H

#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/quadrature.h"
#include "../include/receiver.h"
#include "../include/source.h"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Partial derivates matrices required to compute integrals
 *
 * The matrices are stored in (ispec, iz, ix) format
 *
 */
struct partial_derivatives {
  specfem::kokkos::DeviceView3d<type_real> xix; ///< inverted partial derivates
                                                ///< \f$\partial \xi / \partial
                                                ///< x\f$ stored on the device
  specfem::kokkos::HostMirror3d<type_real> h_xix; ///< inverted partial
                                                  ///< derivates \f$\partial \xi
                                                  ///< / \partial x\f$ stored on
                                                  ///< the host
  specfem::kokkos::DeviceView3d<type_real> xiz; ///< inverted partial derivates
                                                ///< \f$\partial \xi / \partial
                                                ///< z\f$ stored on the device
  specfem::kokkos::HostMirror3d<type_real> h_xiz; ///< inverted partial
                                                  ///< derivates \f$\partial \xi
                                                  ///< / \partial z\f$ stored on
                                                  ///< the host
  specfem::kokkos::DeviceView3d<type_real> gammax;   ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial x\f$
                                                     ///< stored on device
  specfem::kokkos::HostMirror3d<type_real> h_gammax; ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial x\f$
                                                     ///< stored on host
  specfem::kokkos::DeviceView3d<type_real> gammaz;   ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial z\f$
                                                     ///< stored on device
  specfem::kokkos::HostMirror3d<type_real> h_gammaz; ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial z\f$
                                                     ///< stored on host
  specfem::kokkos::DeviceView3d<type_real> jacobian; ///< Jacobian values stored
                                                     ///< on device
  specfem::kokkos::HostMirror3d<type_real> h_jacobian; ///< Jacobian values
                                                       ///< stored on host
  /**
   * @brief Default constructor
   *
   */
  partial_derivatives(){};
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   */
  partial_derivatives(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param coorg (x,z) for every spectral element control node
   * @param knods Global control element number for every control node
   * @param quadx Quadrature object in x dimension
   * @param quadz Quadrature object in z dimension
   */
  partial_derivatives(const specfem::kokkos::HostView2d<type_real> coorg,
                      const specfem::kokkos::HostView2d<int> knods,
                      const specfem::quadrature::quadrature &quadx,
                      const specfem::quadrature::quadrature &quadz);

  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};
/**
 * @brief Material properties stored at every quadrature point
 *
 */
struct properties {
  /**
   * @name Material properties
   *
   * h_ prefixes denote views stored on host
   */
  ///@{
  specfem::kokkos::DeviceView3d<type_real> rho;
  specfem::kokkos::HostMirror3d<type_real> h_rho;

  specfem::kokkos::DeviceView3d<type_real> mu;
  specfem::kokkos::HostMirror3d<type_real> h_mu;

  specfem::kokkos::HostView3d<type_real> kappa;

  specfem::kokkos::HostView3d<type_real> qmu;

  specfem::kokkos::HostView3d<type_real> qkappa;

  specfem::kokkos::HostView3d<type_real> rho_vp;

  specfem::kokkos::HostView3d<type_real> rho_vs;

  specfem::kokkos::DeviceView3d<type_real> lambdaplus2mu;
  specfem::kokkos::HostMirror3d<type_real> h_lambdaplus2mu;
  ///@}
  // element type is defined in config.h
  specfem::kokkos::DeviceView1d<specfem::elements::type>
      ispec_type; ///< type of element
                  ///< stored on device
  specfem::kokkos::HostMirror1d<specfem::elements::type>
      h_ispec_type; ///< type of element
                    ///< stored on host

  /**
   * @brief Default constructor
   *
   */
  properties(){};
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number for spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   */
  properties(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param kmato Material specification number
   * @param materials Pointer to material objects read from database file
   * @param nspec Number for spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   */
  properties(const specfem::kokkos::HostView1d<int> kmato,
             const std::vector<specfem::material *> &materials, const int nspec,
             const int ngllx, const int ngllz);

  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};

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
  specfem::kokkos::DeviceView1d<specfem::forcing_function::stf_storage>
      stf_array; ///< Pointer to source time function for every source stored on
                 ///< device
  specfem::kokkos::HostMirror1d<specfem::forcing_function::stf_storage>
      h_stf_array; ///< Pointer to source time function for every source stored
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
  sources(const std::vector<specfem::sources::source *> &sources,
          const specfem::quadrature::quadrature &quadx,
          const specfem::quadrature::quadrature &quadz, const type_real xmax,
          const type_real xmin, const type_real zmax, const type_real zmin,
          specfem::MPI::MPI *mpi);
  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};

/**
 * @brief This struct is used to store receiver arrays required to interpolate
 * fields during seismogram calculations
 *
 */
struct receivers {
  specfem::kokkos::DeviceView4d<type_real> receiver_array; ///< Array to store
                                                           ///< lagrange
                                                           ///< interpolants for
                                                           ///< receivers on the
                                                           ///< device
  specfem::kokkos::HostMirror4d<type_real> h_receiver_array; ///< Array to store
                                                             ///< lagrange
                                                             ///< interpolants
                                                             ///< for sources
                                                             ///< stored on host
  specfem::kokkos::DeviceView1d<int> ispec_array;   ///< Spectral element number
                                                    ///< where the source lies
                                                    ///< stored on device
  specfem::kokkos::HostMirror1d<int> h_ispec_array; ///< Spectral element number
                                                    ///< where the source lies
                                                    ///< stored on host
  specfem::kokkos::DeviceView1d<type_real> cos_recs; ///< consine of angle used
                                                     ///< to rotate receiver
                                                     ///< components stored on
                                                     ///< device
  specfem::kokkos::HostMirror1d<type_real> h_cos_recs; ///< consine of angle
                                                       ///< used to rotate
                                                       ///< receiver components
                                                       ///< stored on host
  specfem::kokkos::DeviceView1d<type_real> sin_recs; ///< sine of angle used to
                                                     ///< rotate receiver
                                                     ///< components stored on
                                                     ///< device
  specfem::kokkos::HostMirror1d<type_real> h_sin_recs; ///< sine of angle used
                                                       ///< to rotate receiver
                                                       ///< components stored on
                                                       ///< host
  specfem::kokkos::DeviceView5d<type_real> field;      ///< Container to store
                                                  ///< spectral element field
                                                  ///< used in computing
                                                  ///< seismograms stored on the
                                                  ///< device
  specfem::kokkos::DeviceView4d<type_real> seismogram; ///< Container to store
                                                       ///< computed seismograms
                                                       ///< stored on the device
  specfem::kokkos::HostMirror4d<type_real> h_seismogram; ///< Container to store
                                                         ///< computed
                                                         ///< seismograms stored
                                                         ///< on the device
  specfem::kokkos::DeviceView1d<specfem::seismogram::type>
      seismogram_types; ///< Types of seismograms to be calculated stored on the
                        ///< device
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

struct coordinates {

  specfem::kokkos::HostView2d<type_real> coord; ///< (x, z) for every distinct
                                                ///< control node
  /**
   * @name Coodindates meta data
   **/
  ///@{
  type_real xmax; ///< maximum x-coorinate of the quadrature point within this
                  ///< MPI slice
  type_real xmin; ///< minimum x-coorinate of the quadrature point within this
                  ///< MPI slice
  type_real zmax; ///< maximum z-coorinate of the quadrature point within this
                  ///< MPI slice
  type_real zmin; ///< minimum z-coorinate of the quadrature point within this
                  ///< MPI slice
  ///@}
};

struct compute {
  specfem::kokkos::DeviceView3d<int> ibool;   ///< Global number for every
                                              ///< quadrature point stored on
                                              ///< device
  specfem::kokkos::HostMirror3d<int> h_ibool; ///< Global number for every
                                              ///< quadrature point stored on
                                              ///< host
  specfem::compute::coordinates coordinates;  ///< Cartesian coordinates and
                                              ///< related meta-data
  /**
   * @brief Default constructor
   *
   */
  compute(){};
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number for spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   */
  compute(const int nspec, const int ngllx, const int ngllz);
  /**
   * @brief Construct allocate and assign views
   *
   * @param coorg (x_a, z_a) for every control node
   * @param knods Global control element number for every control node
   * @param quadx Quarature object in x dimension
   * @param quadz Quadrature object in z dimension
   */
  compute(const specfem::kokkos::HostView2d<type_real> coorg,
          const specfem::kokkos::HostView2d<int> knods,
          const specfem::quadrature::quadrature &quadx,
          const specfem::quadrature::quadrature &quadz);
  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};

} // namespace compute
} // namespace specfem

#endif
