#include "domain/impl/elements/kernel.hpp"
#include "domain/impl/elements/kernel.tpp"

constexpr static auto forward = specfem::wavefield::simulation_field::forward;
constexpr static auto adjoint = specfem::wavefield::simulation_field::adjoint;
constexpr static auto backward = specfem::wavefield::simulation_field::backward;

constexpr static auto dim2 = specfem::dimension::type::dim2;

constexpr static auto elastic = specfem::element::medium_tag::elastic;
constexpr static auto acoustic = specfem::element::medium_tag::acoustic;

constexpr static auto isotropic = specfem::element::property_tag::isotropic;
constexpr static auto anisotropic = specfem::element::property_tag::anisotropic;

constexpr static auto dirichlet =
    specfem::element::boundary_tag::acoustic_free_surface;
constexpr static auto stacey = specfem::element::boundary_tag::stacey;
constexpr static auto none = specfem::element::boundary_tag::none;
constexpr static auto composite_stacey_dirichlet =
    specfem::element::boundary_tag::composite_stacey_dirichlet;

#define GENERATE_KERNELS(medium_tag, property_tag, ngll)                       \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      forward, dim2, medium_tag, property_tag, none, ngll>;                    \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      adjoint, dim2, medium_tag, property_tag, none, ngll>;                    \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      backward, dim2, medium_tag, property_tag, none, ngll>;                   \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      forward, dim2, medium_tag, property_tag, dirichlet, ngll>;               \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      adjoint, dim2, medium_tag, property_tag, dirichlet, ngll>;               \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      backward, dim2, medium_tag, property_tag, dirichlet, ngll>;              \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      forward, dim2, medium_tag, property_tag, stacey, ngll>;                  \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      adjoint, dim2, medium_tag, property_tag, stacey, ngll>;                  \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      backward, dim2, medium_tag, property_tag, stacey, ngll>;                 \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      forward, dim2, medium_tag, property_tag, composite_stacey_dirichlet,     \
      ngll>;                                                                   \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      adjoint, dim2, medium_tag, property_tag, composite_stacey_dirichlet,     \
      ngll>;                                                                   \
  template class specfem::domain::impl::kernels::element_kernel_base<          \
      backward, dim2, medium_tag, property_tag, composite_stacey_dirichlet,    \
      ngll>;                                                                   \
  template class specfem::domain::impl::kernels::element_kernel<               \
      forward, dim2, medium_tag, property_tag, none, ngll>;                    \
  template class specfem::domain::impl::kernels::element_kernel<               \
      adjoint, dim2, medium_tag, property_tag, none, ngll>;                    \
  template class specfem::domain::impl::kernels::element_kernel<               \
      backward, dim2, medium_tag, property_tag, none, ngll>;                   \
  template class specfem::domain::impl::kernels::element_kernel<               \
      forward, dim2, medium_tag, property_tag, dirichlet, ngll>;               \
  template class specfem::domain::impl::kernels::element_kernel<               \
      adjoint, dim2, medium_tag, property_tag, dirichlet, ngll>;               \
  template class specfem::domain::impl::kernels::element_kernel<               \
      backward, dim2, medium_tag, property_tag, dirichlet, ngll>;              \
  template class specfem::domain::impl::kernels::element_kernel<               \
      forward, dim2, medium_tag, property_tag, stacey, ngll>;                  \
  template class specfem::domain::impl::kernels::element_kernel<               \
      adjoint, dim2, medium_tag, property_tag, stacey, ngll>;                  \
  template class specfem::domain::impl::kernels::element_kernel<               \
      backward, dim2, medium_tag, property_tag, stacey, ngll>;                 \
  template class specfem::domain::impl::kernels::element_kernel<               \
      forward, dim2, medium_tag, property_tag, composite_stacey_dirichlet,     \
      ngll>;                                                                   \
  template class specfem::domain::impl::kernels::element_kernel<               \
      adjoint, dim2, medium_tag, property_tag, composite_stacey_dirichlet,     \
      ngll>;                                                                   \
  template class specfem::domain::impl::kernels::element_kernel<               \
      backward, dim2, medium_tag, property_tag, composite_stacey_dirichlet,    \
      ngll>;

// Explicit template instantiation

GENERATE_KERNELS(elastic, isotropic, 5)

GENERATE_KERNELS(elastic, anisotropic, 5)

GENERATE_KERNELS(acoustic, isotropic, 5)

GENERATE_KERNELS(elastic, isotropic, 8)

GENERATE_KERNELS(elastic, anisotropic, 8)

GENERATE_KERNELS(acoustic, isotropic, 8)
