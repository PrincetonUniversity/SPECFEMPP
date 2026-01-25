#include "compute_coupling.hpp"
#include "compute_coupling.tpp"
#include "enumerations/interface.hpp"
#include "specfem/macros.hpp"
#include <type_traits>

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
     INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
     BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                  COMPOSITE_STACEY_DIRICHLET)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template void specfem::compute::impl::compute_coupling,
         (_DIMENSION_TAG_, _CONNECTION_TAG_,
          specfem::simulation::field_type::forward, 5, 5, _INTERFACE_TAG_,
          _BOUNDARY_TAG_, specfem::interface::flux_scheme_tag::natural),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::compute::impl::compute_coupling,
         (_DIMENSION_TAG_, _CONNECTION_TAG_,
          specfem::simulation::field_type::backward, 5, 5, _INTERFACE_TAG_,
          _BOUNDARY_TAG_, specfem::interface::flux_scheme_tag::natural),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::compute::impl::compute_coupling,
         (_DIMENSION_TAG_, _CONNECTION_TAG_,
          specfem::simulation::field_type::adjoint, 5, 5, _INTERFACE_TAG_,
          _BOUNDARY_TAG_, specfem::interface::flux_scheme_tag::natural),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        /** instantiation for NGLL = 8     */
        (template void specfem::compute::impl::compute_coupling,
         (_DIMENSION_TAG_, _CONNECTION_TAG_,
          specfem::simulation::field_type::forward, 8, 8, _INTERFACE_TAG_,
          _BOUNDARY_TAG_, specfem::interface::flux_scheme_tag::natural),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::compute::impl::compute_coupling,
         (_DIMENSION_TAG_, _CONNECTION_TAG_,
          specfem::simulation::field_type::backward, 8, 8, _INTERFACE_TAG_,
          _BOUNDARY_TAG_, specfem::interface::flux_scheme_tag::natural),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::compute::impl::compute_coupling,
         (_DIMENSION_TAG_, _CONNECTION_TAG_,
          specfem::simulation::field_type::adjoint, 8, 8, _INTERFACE_TAG_,
          _BOUNDARY_TAG_, specfem::interface::flux_scheme_tag::natural),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);)))
