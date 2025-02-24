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
      : assembly(assembly), coupling_interfaces_elastic(assembly),
        coupling_interfaces_acoustic(assembly) {}

  template <specfem::element::medium_tag medium>
  inline void update_wavefields(const int istep) {

#define CALL_COUPLING_INTERFACES_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)           \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    CREATE_VARIABLE_NAME(coupling_interfaces, GET_NAME(MEDIUM_TAG))            \
        .compute_coupling();                                                   \
  }

    CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
        CALL_COUPLING_INTERFACES_FUNCTION,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef CALL_COUPLING_INTERFACES_FUNCTION

#define CALL_SOURCE_FORCE_UPDATE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,      \
                                 BOUNDARY_TAG)                                 \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    impl::compute_source_interaction<                                          \
        dimension, wavefield, ngll, GET_TAG(MEDIUM_TAG),                       \
        GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(assembly, istep);        \
  }

    CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
        CALL_SOURCE_FORCE_UPDATE,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
                    WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                          BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                          BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_SOURCE_FORCE_UPDATE

#define CALL_STIFFNESS_FORCE_UPDATE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,   \
                                    BOUNDARY_TAG)                              \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    impl::compute_stiffness_interaction<                                       \
        dimension, wavefield, ngll, GET_TAG(MEDIUM_TAG),                       \
        GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(assembly, istep);        \
  }

    CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
        CALL_STIFFNESS_FORCE_UPDATE,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
                    WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                          BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                          BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_STIFFNESS_FORCE_UPDATE

#define CALL_DIVIDE_MASS_MATRIX_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)            \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    impl::divide_mass_matrix<dimension, wavefield, GET_TAG(MEDIUM_TAG)>(       \
        assembly);                                                             \
  }

    CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
        CALL_DIVIDE_MASS_MATRIX_FUNCTION,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef CALL_DIVIDE_MASS_MATRIX_FUNCTION
  }

  void initialize(const type_real &dt) {

#define CALL_COMPUTE_MASS_MATRIX_FUNCTION(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG, BOUNDARY_TAG)          \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG)) {                         \
    impl::compute_mass_matrix<dimension, wavefield, ngll, GET_TAG(MEDIUM_TAG), \
                              GET_TAG(PROPERTY_TAG), GET_TAG(BOUNDARY_TAG)>(   \
        dt, assembly);                                                         \
  }

    CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
        CALL_COMPUTE_MASS_MATRIX_FUNCTION,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
                    WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                          BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                          BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_COMPUTE_MASS_MATRIX_FUNCTION

#define CALL_INITIALIZE_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)                    \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG)) {                         \
    impl::invert_mass_matrix<dimension, wavefield, GET_TAG(MEDIUM_TAG)>(       \
        assembly);                                                             \
  }

    CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
        CALL_INITIALIZE_FUNCTION,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef CALL_INITIALIZE_FUNCTION

    return;
  }

  inline void compute_seismograms(const int &isig_step) {

#define CALL_COMPUTE_SEISMOGRAMS_FUNCTION(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG)                        \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG)) {                         \
    impl::compute_seismograms<dimension, wavefield, ngll, GET_TAG(MEDIUM_TAG), \
                              GET_TAG(PROPERTY_TAG)>(assembly, isig_step);     \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        CALL_COMPUTE_SEISMOGRAMS_FUNCTION,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef CALL_COMPUTE_SEISMOGRAMS_FUNCTION
  }

private:
  specfem::compute::assembly assembly;
#define COUPLING_INTERFACES_DECLARATION(DIMENSION_TAG, MEDIUM_TAG)             \
  impl::interface_kernels<WavefieldType, GET_TAG(DIMENSION_TAG),               \
                          GET_TAG(MEDIUM_TAG)>                                 \
      CREATE_VARIABLE_NAME(coupling_interfaces, GET_NAME(MEDIUM_TAG));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(COUPLING_INTERFACES_DECLARATION,
                                 WHERE(DIMENSION_TAG_DIM2)
                                     WHERE(MEDIUM_TAG_ELASTIC_SV,
                                           MEDIUM_TAG_ACOUSTIC))

#undef COUPLING_INTERFACES_DECLARATION
};

} // namespace kokkos_kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_HPP */
