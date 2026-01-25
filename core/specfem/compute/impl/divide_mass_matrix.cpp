#include "divide_mass_matrix.hpp"
#include "divide_mass_matrix.tpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
    INSTANTIATE(
        (template void specfem::compute::impl::divide_mass_matrix,
         (_DIMENSION_TAG_, specfem::simulation::field_type::forward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::compute::impl::divide_mass_matrix,
         (_DIMENSION_TAG_, specfem::simulation::field_type::backward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);),
        (template void specfem::compute::impl::divide_mass_matrix,
         (_DIMENSION_TAG_, specfem::simulation::field_type::adjoint,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim2>
              &);)))

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM3), MEDIUM_TAG(ACOUSTIC, ELASTIC)),
    INSTANTIATE(
        (template void specfem::compute::impl::divide_mass_matrix,
         (_DIMENSION_TAG_, specfem::simulation::field_type::forward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim3>
              &);),
        (template void specfem::compute::impl::divide_mass_matrix,
         (_DIMENSION_TAG_, specfem::simulation::field_type::backward,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim3>
              &);),
        (template void specfem::compute::impl::divide_mass_matrix,
         (_DIMENSION_TAG_, specfem::simulation::field_type::adjoint,
          _MEDIUM_TAG_),
         (const specfem::assembly::assembly<specfem::dimension::type::dim3>
              &);)))
