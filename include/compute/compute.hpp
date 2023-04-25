#ifndef _COMPUTE_HPP
#define _COMPUTE_HPP

#include "kokkos_abstractions.h"
#include "quadrature.h"
#include "receiver.h"
#include "source.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {
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
