#include "update_wavefields.hpp"
#include "enumerations/interface.hpp"
#include "impl/compute_coupling.hpp"
#include "impl/compute_coupling.tpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_source_interaction.tpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/compute_stiffness_interaction.tpp"
#include "impl/divide_mass_matrix.hpp"
#include "impl/divide_mass_matrix.tpp"
#include "specfem/assembly.hpp"
#include "specfem/macros.hpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::forward, _DIMENSION_TAG_, 5,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int);),
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::backward, _DIMENSION_TAG_, 5,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int);),
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::adjoint, _DIMENSION_TAG_, 5,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int);),
        /** instantiation for NGLL = 8     */
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::forward, _DIMENSION_TAG_, 8,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int);),
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::backward, _DIMENSION_TAG_, 8,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int);),
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::adjoint, _DIMENSION_TAG_, 8,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim2> &,
          const int);)))

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC),
     PROPERTY_TAG(ISOTROPIC), BOUNDARY_TAG(NONE)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template int specfem::compute::update_wavefields,
         (specfem::simulation::field_type::forward, _DIMENSION_TAG_, 5,
          _MEDIUM_TAG_),
         (specfem::assembly::assembly<specfem::dimension::type::dim3> &,
          const int);)))
