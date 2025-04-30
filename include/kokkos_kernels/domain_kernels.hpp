#ifndef _SPECFEM_KERNELS_HPP
#define _SPECFEM_KERNELS_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "impl/compute_mass_matrix.hpp"
#include "impl/compute_seismogram.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "impl/interface_kernels.hpp"
#include "impl/invert_mass_matrix.hpp"

namespace specfem {
namespace kokkos_kernels {
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, int NGLL>
class domain_kernels {
public:
  constexpr static auto dimension = DimensionType;
  constexpr static auto wavefield = WavefieldType;
  constexpr static auto ngll = NGLL;

  domain_kernels(const specfem::compute::assembly &assembly)
      : assembly(assembly), coupling_interfaces_dim2_elastic_psv(assembly),
        coupling_interfaces_dim2_acoustic(assembly) {}

  template <specfem::element::medium_tag medium>
  inline int update_wavefields(const int istep) {

    int elements_updated = 0;

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ACOUSTIC)),
        CAPTURE(coupling_interfaces) {
          if constexpr (dimension == _dimension_tag_ &&
                        medium == _medium_tag_) {
            _coupling_interfaces_.compute_coupling();
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          if constexpr (dimension == _dimension_tag_ &&
                        medium == _medium_tag_) {
            impl::compute_source_interaction<dimension, wavefield, ngll,
                                             _medium_tag_, _property_tag_,
                                             _boundary_tag_>(assembly, istep);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          if constexpr (dimension == _dimension_tag_ &&
                        medium == _medium_tag_) {
            elements_updated += impl::compute_stiffness_interaction<
                dimension, wavefield, ngll, _medium_tag_, _property_tag_,
                _boundary_tag_>(assembly, istep);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        {
          if constexpr (dimension == _dimension_tag_ &&
                        medium == _medium_tag_) {
            impl::divide_mass_matrix<dimension, wavefield, _medium_tag_>(
                assembly);
          }
        })

    return elements_updated;
  }

  void initialize(const type_real &dt) {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          if constexpr (dimension == _dimension_tag_) {
            impl::compute_mass_matrix<dimension, wavefield, ngll, _medium_tag_,
                                      _property_tag_, _boundary_tag_>(dt,
                                                                      assembly);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        {
          if constexpr (dimension == _dimension_tag_) {
            impl::invert_mass_matrix<dimension, wavefield, _medium_tag_>(
                assembly);
          }
        })

    return;
  }

  inline void compute_seismograms(const int &isig_step) {

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        {
          if constexpr (dimension == _dimension_tag_) {
            impl::compute_seismograms<dimension, wavefield, ngll, _medium_tag_,
                                      _property_tag_>(assembly, isig_step);
          }
        })
  }

private:
  specfem::compute::assembly assembly;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ACOUSTIC)),
                      DECLARE(((impl::interface_kernels,
                                (WavefieldType, _DIMENSION_TAG_, _MEDIUM_TAG_)),
                               coupling_interfaces)))
};

} // namespace kokkos_kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_HPP */
