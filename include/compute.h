#ifndef COMPUTE_H
#define COMPUTE_H

#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/quadrature.h"
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
  specfem::HostView3d<type_real> xix;    ///< inverted partial derivates
                                         ///< \f$\partial \xi / \partial x\f$
  specfem::HostView3d<type_real> xiz;    ///< inverted partial derivates
                                         ///< \f$\partial \xi / \partial z\f$
  specfem::HostView3d<type_real> gammax; ///< inverted partial derivates
                                         ///< \f$\partial \gamma / \partial x\f$
  specfem::HostView3d<type_real> gammaz; ///< inverted partial derivates
                                         ///< \f$\partial \gamma / \partial z\f$
  specfem::HostView3d<type_real> jacobian; ///< Jacobian values
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
  partial_derivatives(const specfem::HostView2d<type_real> coorg,
                      const specfem::HostView2d<int> knods,
                      const specfem::quadrature::quadrature &quadx,
                      const specfem::quadrature::quadrature &quadz);
};
/**
 * @brief Material properties stored at every quadrature point
 *
 */
struct properties {
  /**
   * @name Material properties
   *
   */
  ///@{
  specfem::HostView3d<type_real> rho, mu, kappa, qmu, qkappa, rho_vp, rho_vs,
      lambdaplus2mu;
  ///@}
  // element type is defined in config.h
  specfem::HostView1d<element_type> ispec_type; ///< type of element. Available
                                                ///< element types are defined
                                                ///< in config.h

  /**
   * @brief Default
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
  properties(const specfem::HostView1d<int> kmato,
             const std::vector<specfem::material *> &materials, const int nspec,
             const int ngllx, const int ngllz);
};

/**
 * @brief This struct is used to store arrays required to impose source during
 * the time loop
 *
 * @note Does not implement moving sources yet
 *
 */
struct sources {
  specfem::HostView4d<type_real> source_array; ///< Array to store lagrange
                                               ///< interpolants for sources.
                                               ///< These arrays are used to
                                               ///< impose source effects at end
                                               ///< of every time-step.
  specfem::HostView2d<type_real> stf_array; //< Value of source-time function at
                                            ///< every time step
  specfem::HostView1d<type_real> ispec_array; ///< Spectral element number where
                                              ///< the source lies
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
  sources(std::vector<specfem::sources::source *> sources,
          specfem::quadrature::quadrature &quadx,
          specfem::quadrature::quadrature &quadz,
          specfem::TimeScheme::TimeScheme *it, specfem::MPI::MPI *mpi);
};

// /**
//  * @brief This struct is used to store arrays required to impose adjoint
//  sources (at reciever locations) during
//  * the time loop
//  *
//  * @note Does not implement moving sources yet
//  *
//  */
// struct recievers {
//   specfem::HostView4d<type_real> reciever_array; ///< Array to store lagrange
//                                                ///< interpolants for sources.
//                                                ///< These arrays are used to
//                                                ///< impose reciever effects
//                                                at end
//                                                ///< of every time-step.
//   spefecm::HostView2d<type_real> stf_array; //< Value of source-time function
//   at
//                                             ///< every time step
//   /**
//    * @brief Default constructor
//    *
//    */
//   recievers(){};
//   /**
//    * @brief Constructor to allocate and assign views
//    *
//    * @param recievers Pointer to recievers objects read from sources file
//    * @param quadx Quarature object in x dimension
//    * @param quadz Quadrature object in z dimension
//    * @param mpi Pointer to the MPI object
//    */
//   recievers(std::vector<specfem::sources::source *> recievers,
//           specfem::quadrature::quadrature &quadx,
//           specfem::quadrature::quadrature &quadz, specfem::MPI::MPI *mpi);
// }

struct coordinates {

  specfem::HostView2d<type_real> coord; ///< (x, z) for every distinct control
                                        ///< node
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
  specfem::HostView3d<int> ibool; ///< Global number for every quadrature point
  specfem::compute::coordinates coordinates; ///< Cartesian coordinates and
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
  compute(const specfem::HostView2d<type_real> coorg,
          const specfem::HostView2d<int> knods,
          const specfem::quadrature::quadrature &quadx,
          const specfem::quadrature::quadrature &quadz);
};

} // namespace compute
} // namespace specfem

#endif
