#ifndef _DOMAIN_KERNELS_HPP
#define _DOMAIN_KERNELS_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/interface.hpp"
#include "domain/impl/receivers/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "enumerations/boundary_conditions/none.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

/**
 * @brief Kernels object used to compute elemental kernels
 *
 * This object consists of various Kokkos kernels used to compute elemental
 * contributions from different types of element, sources, and receivers.
 *
 * The template parameters are inherited from the domain class.
 *
 * @tparam medium class defining the domain medium. Separate implementations
 * exist for elastic, acoustic or poroelastic media
 * @tparam qp_type class used to define the quadrature points either
 * at compile time or run time
 */
template <class medium, class qp_type> class kernels {

public:
  using dimension = specfem::enums::element::dimension::dim2; // Dimension of
                                                              // the domain
  using medium_type = medium; // Type of medium i.e. acoustic, elastic or
                              // poroelastic
  using quadrature_point_type = qp_type; // Type of quadrature points i.e.
                                         // static or dynamic

  /**
   * @brief Default constructor
   *
   */
  kernels() = default;

  /**
   * @brief Construct a new kernels object
   *
   * @param ibool Global index for every GLL point in the SPECFEM simulation
   * @param partial_derivatives struct used to store partial derivatives at
   * every GLL point
   * @param properties struct used to store material properties at every GLL
   * point
   * @param sources struct used to store information about sources
   * @param receives struct used to store information about receivers
   * @param quadx Pointer to quadrature points in X direction
   * @param quadz Pointer to quadrature points in Z direction
   * @param quadrature_points quadrature points object to define number of
   * quadrature points either at compile time or run time
   * @param field wavefield inside the domain
   * @param field_dot derivative of wavefield inside the domain
   * @param field_dot_dot double derivative of wavefield inside the domain
   * @param mass_matrix mass matrix for every GLL point inside the domain
   */
  kernels(const specfem::compute::assembly &assembly,
          const quadrature_point_type &quadrature_points);

  // /**
  //  * @brief
  //  *
  //  */
  // template <specfem::enums::time_scheme::type time_scheme>
  // inline void mass_time_contribution(const type_real &dt) const {
  //   isotropic_elements.template mass_time_contribution<time_scheme>(dt);
  //   isotropic_elements_dirichlet.template
  //   mass_time_contribution<time_scheme>(
  //       dt);
  //   isotropic_elements_stacey.template
  //   mass_time_contribution<time_scheme>(dt);
  //   isotropic_elements_stacey_dirichlet
  //       .template mass_time_contribution<time_scheme>(dt);
  //   return;
  // }

  // /**
  //  * @brief execute Kokkos kernel to compute contribution of stiffness matrix
  //  to
  //  * the global acceleration
  //  *
  //  */
  // inline void compute_stiffness_interaction() const {
  //   isotropic_elements.compute_stiffness_interaction();
  //   isotropic_elements_dirichlet.compute_stiffness_interaction();
  //   isotropic_elements_stacey.compute_stiffness_interaction();
  //   isotropic_elements_stacey_dirichlet.compute_stiffness_interaction();
  //   return;
  // }

  /**
   * @brief execute Kokkos kernel to compute the mass matrix for every GLL point
   *
   */
  inline void compute_mass_matrix() const {
    isotropic_elements.compute_mass_matrix();
    // isotropic_elements_dirichlet.compute_mass_matrix();
    // isotropic_elements_stacey.compute_mass_matrix();
    // isotropic_elements_stacey_dirichlet.compute_mass_matrix();
    return;
  }

  // /**
  //  * @brief execute Kokkos kernel compute the contribution of sources to the
  //  * global acceleration
  //  *
  //  * @param timeval time value at the current time step
  //  */
  // inline void compute_source_interaction(const type_real timeval) const {
  //   isotropic_sources.compute_source_interaction(timeval);
  //   return;
  // }

  // /**
  //  * @brief execute Kokkos kernel to compute seismogram values at every
  //  receiver
  //  * for the current seismogram step
  //  *
  //  * A seismogram step is defined as the current time step divided by the
  //  * seismogram sampling rate
  //  *
  //  * @param isig_step current seismogram step.
  //  */
  // inline void compute_seismograms(const int &isig_step) const {
  //   isotropic_receivers.compute_seismograms(isig_step);
  //   return;
  // }

private:
  // template <class property>
  // using dirichlet = specfem::enums::boundary_conditions::template dirichlet<
  //     dimension, medium_type, property, quadrature_point_type>; // Dirichlet
  //                                                               // boundary
  //                                                               // conditions

  // template <class property>
  // using stacey = specfem::enums::boundary_conditions::template stacey<
  //     dimension, medium_type, property, quadrature_point_type>; // Stacey
  //                                                               // boundary
  //                                                               // conditions

  template <class property>
  using none = specfem::enums::boundary_conditions::template none<
      dimension, medium_type, property, quadrature_point_type>; // No boundary
                                                                // conditions

  // template <class BC1, class BC2>
  // using composite_boundary =
  //     specfem::enums::boundary_conditions::composite_boundary<
  //         BC1, BC2>; // Composite boundary conditions

  /**
   * @brief Elemental kernels for isotropic elements
   *
   */
  specfem::domain::impl::kernels::element_kernel<
      medium_type, quadrature_point_type,
      specfem::enums::element::property::isotropic,
      none<specfem::enums::element::property::isotropic> >
      isotropic_elements;

  // /**
  //  * @brief Elemental kernels for isotropic elements with dirichlet boundary
  //  * conditions
  //  *
  //  */
  // specfem::domain::impl::kernels::element_kernel<
  //     medium_type, quadrature_point_type,
  //     specfem::enums::element::property::isotropic,
  //     dirichlet<specfem::enums::element::property::isotropic> >
  //     isotropic_elements_dirichlet;

  // /**
  //  * @brief Elemental kernels for isotropic elements with stacey boundary
  //  *
  //  */
  // specfem::domain::impl::kernels::element_kernel<
  //     medium_type, quadrature_point_type,
  //     specfem::enums::element::property::isotropic,
  //     stacey<specfem::enums::element::property::isotropic> >
  //     isotropic_elements_stacey;

  // /**
  //  * @brief Elemental kernels for isotropic elements with composite stacey
  //  and
  //  * dirichlet boundary
  //  *
  //  */
  // specfem::domain::impl::kernels::element_kernel<
  //     medium_type, quadrature_point_type,
  //     specfem::enums::element::property::isotropic,
  //     composite_boundary<
  //         stacey<specfem::enums::element::property::isotropic>,
  //         dirichlet<specfem::enums::element::property::isotropic> > >
  //     isotropic_elements_stacey_dirichlet;

  // /**
  //  * @brief Elemental source kernels for isotropic elements
  //  *
  //  */
  // specfem::domain::impl::kernels::source_kernel<
  //     medium_type, quadrature_point_type,
  //     specfem::enums::element::property::isotropic>
  //     isotropic_sources;

  // /**
  //  * @brief Elemental receiver kernels for isotropic elements
  //  *
  //  */
  // specfem::domain::impl::kernels::receiver_kernel<
  //     medium_type, quadrature_point_type,
  //     specfem::enums::element::property::isotropic>
  //     isotropic_receivers; ///< Elemental receiver kernels for isotropic
  //                          ///< elements
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_KERNELS_HPP
