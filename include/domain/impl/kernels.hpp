#pragma once

#include "compute/interface.hpp"
#include "domain/impl/elements/kernel.hpp"
#include "domain/impl/receivers/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium, typename qp_type>
class kernels {

public:
  using quadrature_point_type = qp_type;
  using dimension = specfem::dimension::dimension<DimensionType>;
  kernels() = default;

  kernels(const type_real dt, const specfem::compute::assembly &assembly,
          const quadrature_point_type &quadrature_points);

  /**
   * @brief Compute the interaction of stiffness matrix with wavefield at a time
   * step
   *
   * @param istep Time step
   */
  inline void compute_stiffness_interaction(const int istep) const {
    isotropic_elements.compute_stiffness_interaction(istep);
    isotropic_elements_dirichlet.compute_stiffness_interaction(istep);
    isotropic_elements_stacey.compute_stiffness_interaction(istep);
    isotropic_elements_stacey_dirichlet.compute_stiffness_interaction(istep);
    return;
  }

  /**
   * @brief Compute the mass matrix
   *
   * @param dt Time step
   */
  inline void compute_mass_matrix(const type_real dt) const {
    isotropic_elements.compute_mass_matrix(dt);
    isotropic_elements_dirichlet.compute_mass_matrix(dt);
    isotropic_elements_stacey.compute_mass_matrix(dt);
    isotropic_elements_stacey_dirichlet.compute_mass_matrix(dt);
    return;
  }

  /**
   * @brief Compute the interaction of source with wavefield at a time step
   *
   * @param timestep Time step
   */
  inline void compute_source_interaction(const int timestep) const {
    isotropic_sources.compute_source_interaction(timestep);
    return;
  }

  /**
   * @brief Compute seismograms at a time step
   *
   * @param isig_step Seismogram time step. Same if seismogram is computed at
   * every time step.
   */
  inline void compute_seismograms(const int &isig_step) const {
    isotropic_receivers.compute_seismograms(isig_step);
    return;
  }

private:
  constexpr static specfem::element::boundary_tag dirichlet =
      specfem::element::boundary_tag::acoustic_free_surface;
  constexpr static specfem::element::boundary_tag stacey =
      specfem::element::boundary_tag::stacey;
  constexpr static specfem::element::boundary_tag none =
      specfem::element::boundary_tag::none;
  constexpr static specfem::element::boundary_tag composite_stacey_dirichlet =
      specfem::element::boundary_tag::composite_stacey_dirichlet;
  constexpr static specfem::element::property_tag isotropic =
      specfem::element::property_tag::isotropic;

  constexpr static int NGLL = quadrature_point_type::NGLL;

  template <specfem::dimension::type dimension,
            specfem::element::property_tag property,
            specfem::element::boundary_tag boundary>
  using element_kernel = specfem::domain::impl::kernels::element_kernel<
      WavefieldType, DimensionType, medium, property, boundary,
      NGLL>; ///< Underlying element kernel data structure

  template <specfem::dimension::type dimension,
            specfem::element::property_tag property>
  using source_kernel = specfem::domain::impl::kernels::source_kernel<
      WavefieldType, DimensionType, medium, property,
      quadrature_point_type>; ///< Underlying source kernel data structure

  template <specfem::dimension::type dimension,
            specfem::element::property_tag property>
  using receiver_kernel = specfem::domain::impl::kernels::receiver_kernel<
      WavefieldType, DimensionType, medium, property,
      quadrature_point_type>; ///< Underlying receiver kernel data structure

  element_kernel<DimensionType, isotropic, none>
      isotropic_elements; ///< Stiffness kernels for isotropic elements

  element_kernel<DimensionType, isotropic, dirichlet>
      isotropic_elements_dirichlet; ///< Stiffness kernels for isotropic
                                    ///< elements with Dirichlet boundary
                                    ///< conditions

  element_kernel<DimensionType, isotropic, stacey>
      isotropic_elements_stacey; ///< Stiffness kernels for isotropic elements
                                 ///< with Stacey boundary conditions

  element_kernel<DimensionType, isotropic, composite_stacey_dirichlet>
      isotropic_elements_stacey_dirichlet; ///< Stiffness kernels for isotropic
                                           ///< elements with Stacey and
                                           ///< Dirichlet boundary conditions on
                                           ///< the same element

  source_kernel<DimensionType, isotropic> isotropic_sources; ///< Source kernels
                                                             ///< for isotropic
                                                             ///< elements

  receiver_kernel<DimensionType, isotropic>
      isotropic_receivers; ///< Kernels for computing seismograms within
                           ///< isotropic elements
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem
