#ifndef _DOMAIN_KERNELS_HPP
#define _DOMAIN_KERNELS_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/interface.hpp"
#include "domain/impl/receivers/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <specfem::simulation::type simulation,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag medium, typename qp_type>
class kernels {

public:
  using quadrature_point_type = qp_type;
  using dimension = specfem::dimension::dimension<DimensionType>;
  kernels() = default;

  kernels(const specfem::compute::assembly &assembly,
          const quadrature_point_type &quadrature_points);

  template <specfem::enums::time_scheme::type time_scheme>
  inline void mass_time_contribution(const type_real &dt) const {
    isotropic_elements.template mass_time_contribution<time_scheme>(dt);
    isotropic_elements_dirichlet.template mass_time_contribution<time_scheme>(
        dt);
    isotropic_elements_stacey.template mass_time_contribution<time_scheme>(dt);
    isotropic_elements_stacey_dirichlet
        .template mass_time_contribution<time_scheme>(dt);
    return;
  }

  inline void compute_stiffness_interaction() const {
    isotropic_elements.compute_stiffness_interaction();
    isotropic_elements_dirichlet.compute_stiffness_interaction();
    isotropic_elements_stacey.compute_stiffness_interaction();
    isotropic_elements_stacey_dirichlet.compute_stiffness_interaction();
    return;
  }

  inline void compute_mass_matrix() const {
    isotropic_elements.compute_mass_matrix();
    isotropic_elements_dirichlet.compute_mass_matrix();
    isotropic_elements_stacey.compute_mass_matrix();
    isotropic_elements_stacey_dirichlet.compute_mass_matrix();
    return;
  }

  inline void compute_source_interaction(const type_real timeval) const {
    isotropic_sources.compute_source_interaction(timeval);
    return;
  }

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

  template <specfem::dimension::type dimension,
            specfem::element::property_tag property,
            specfem::element::boundary_tag boundary>
  using element_kernel = specfem::domain::impl::kernels::element_kernel<
      DimensionType, medium, property, boundary, quadrature_point_type>;

  template <specfem::dimension::type dimension,
            specfem::element::property_tag property>
  using source_kernel = specfem::domain::impl::kernels::source_kernel<
      DimensionType, medium, property, quadrature_point_type>;

  template <specfem::dimension::type dimension,
            specfem::element::property_tag property>
  using receiver_kernel = specfem::domain::impl::kernels::receiver_kernel<
      DimensionType, medium, property, quadrature_point_type>;

  element_kernel<DimensionType, isotropic, none> isotropic_elements;

  element_kernel<DimensionType, isotropic, dirichlet>
      isotropic_elements_dirichlet;

  element_kernel<DimensionType, isotropic, stacey> isotropic_elements_stacey;

  element_kernel<DimensionType, isotropic, composite_stacey_dirichlet>
      isotropic_elements_stacey_dirichlet;

  source_kernel<DimensionType, isotropic> isotropic_sources;

  receiver_kernel<DimensionType, isotropic> isotropic_receivers;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_KERNELS_HPP
