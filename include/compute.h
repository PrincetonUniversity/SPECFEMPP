#ifndef COMPUTE_H
#define COMPUTE_H

#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Matrices required to compute integrals
 *
 * The matrices are stored in (ispec, iz, ix) format
 *
 */
struct coordinates {
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
  coordinates(){};
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   */
  coordinates(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param coorg (x,z) for every spectral element control node
   * @param knods Global control element number for every control node
   * @param quadx Quadrature object in x dimension
   * @param quadz Quadrature object in z dimension
   */
  coordinates(const specfem::HostView2d<type_real> coorg,
              const specfem::HostView2d<int> knods,
              const quadrature::quadrature &quadx,
              const quadrature::quadrature &quadz);
};

struct properties {
  /**
   * Material properties defined for every quadrature point
   *
   */
  specfem::HostView3d<type_real> rho, mu, kappa, qmu, qkappa, rho_vp, rho_vs;
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

struct compute {
  specfem::HostView3d<int> ibool; ///< Global number for every quadrature point
  specfem::HostView2d<type_real> coord; ///< (x, z) for every distinct control
                                        ///< node
  specfem::compute::coordinates coordinates; ///< Matrices required to compute
                                             ///< integrals
  specfem::compute::properties properties; ///< Material properties at elemental
                                           ///< level
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
   * @brief Construct a allocate and assign views
   *
   * @param coorg (x_a, z_a) for every control node
   * @param knods Global control element number for every control node
   * @param kmato Material specification number
   * @param quadx Quarature object in x dimension
   * @param quadz Quadrature object in z dimension
   * @param materials Pointer to material objects read from database file
   */
  compute(const specfem::HostView2d<type_real> coorg,
          const specfem::HostView2d<int> knods,
          const specfem::HostView1d<int> kmato,
          const quadrature::quadrature &quadx,
          const quadrature::quadrature &quadz,
          const std::vector<specfem::material *> &materials);
};

} // namespace compute
} // namespace specfem

#endif
