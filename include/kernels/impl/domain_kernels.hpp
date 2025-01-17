#ifndef _SPECFEM_KERNELS_IMPL_DOMAIN_KERNELS_HPP
#define _SPECFEM_KERNELS_IMPL_DOMAIN_KERNELS_HPP

#include "compute_mass_matrix.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kernels.hpp"

namespace specfem {
namespace kernels {
namespace impl {
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, int NGLL>
class domain_kernels {
public:
  constexpr static auto dimension = DimensionType;
  constexpr static auto wavefield = WavefieldType;
  constexpr static auto ngll = NGLL;

  domain_kernels(const type_real dt, const specfem::compute::assembly &assembly)
      : assembly(assembly), coupling_interfaces_elastic(assembly),
        coupling_interfaces_acoustic(assembly) {}

  void prepare_wavefields();

  template <specfem::element::medium_tag medium>
  inline void update_wavefields(const int istep) {

#define CALL_COUPLING_INTERFACES_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)           \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    CREATE_VARIABLE_NAME(coupling_interfaces, GET_NAME(MEDIUM_TAG))            \
        .compute_coupling();                                                   \
  }

    CALL_MACRO_FOR_ALL_MEDIUM_TAGS(CALL_COUPLING_INTERFACES_FUNCTION,
                                   WHERE(DIMENSION_TAG_DIM2) WHERE(
                                       MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef CALL_COUPLING_INTERFACES_FUNCTION

#define CALL_SOURCE_FORCE_UPDATE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,      \
                                 BOUNDARY_TAG)                                 \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    compute_source_interaction<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG),     \
                               GET_TAG(BOUNDARY_TAG)>(istep);                  \
  }

    CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
        CALL_SOURCE_FORCE_UPDATE,
        WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
                WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                      BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                      BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_SOURCE_FORCE_UPDATE

#define CALL_STIFFNESS_FORCE_UPDATE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,   \
                                    BOUNDARY_TAG)                              \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    compute_stiffness_interaction<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG),  \
                                  GET_TAG(BOUNDARY_TAG)>(istep);               \
  }

    CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
        CALL_STIFFNESS_FORCE_UPDATE,
        WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
                WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                      BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                      BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_STIFFNESS_FORCE_UPDATE

#define CALL_DIVIDE_MASS_MATRIX_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)            \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG) &&                         \
                medium == GET_TAG(MEDIUM_TAG)) {                               \
    divide_mass_matrix<GET_TAG(MEDIUM_TAG)>();                                 \
  }

    CALL_MACRO_FOR_ALL_MEDIUM_TAGS(CALL_DIVIDE_MASS_MATRIX_FUNCTION,
                                   WHERE(DIMENSION_TAG_DIM2) WHERE(
                                       MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

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
        WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
                WHERE(BOUNDARY_TAG_STACEY, BOUNDARY_TAG_NONE,
                      BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                      BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef CALL_COMPUTE_MASS_MATRIX_FUNCTION

#define CALL_INITIALIZE_FUNCTION(DIMENSION_TAG, MEDIUM_TAG)                    \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG)) {                         \
    invert_mass_matrix<GET_TAG(MEDIUM_TAG)>(dt);                               \
  }

    CALL_MACRO_FOR_ALL_MEDIUM_TAGS(CALL_INITIALIZE_FUNCTION,
                                   WHERE(DIMENSION_TAG_DIM2) WHERE(
                                       MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef CALL_INITIALIZE_FUNCTION

    return;
  }

  inline void compute_seismograms(const int &isig_step) {

#define CALL_COMPUTE_SEISMOGRAMS_FUNCTION(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG)                        \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG)) {                         \
    compute_seismograms<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(           \
        isig_step);                                                            \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        CALL_COMPUTE_SEISMOGRAMS_FUNCTION,
        WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef CALL_COMPUTE_SEISMOGRAMS_FUNCTION
  }

private:
  specfem::compute::assembly assembly;
#define COUPLING_INTERFACES_DECLARATION(DIMENSION_TAG, MEDIUM_TAG)             \
  specfem::kernels::impl::interface_kernels<                                   \
      WavefieldType, GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG)>              \
      CREATE_VARIABLE_NAME(coupling_interfaces, GET_NAME(MEDIUM_TAG));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(COUPLING_INTERFACES_DECLARATION,
                                 WHERE(DIMENSION_TAG_DIM2)
                                     WHERE(MEDIUM_TAG_ELASTIC,
                                           MEDIUM_TAG_ACOUSTIC))

#undef COUPLING_INTERFACES_DECLARATION

  // specfem::kernels::impl::kernels<WavefieldType, DimensionType,
  //                                 specfem::element::medium_tag::elastic,
  //                                 qp_type>
  //     elastic_kernels;
  // specfem::kernels::impl::kernels<WavefieldType, DimensionType,
  //                                 specfem::element::medium_tag::acoustic,
  //                                 qp_type>
  //     acoustic_kernels;

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::boundary_tag BoundaryTag>
  void compute_stiffness_interaction(const int istep);

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::boundary_tag BoundaryTag>
  void compute_source_interaction(const int istep);

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  void compute_seismograms(const int &isig_step);

  template <specfem::element::medium_tag MediumTag> void divide_mass_matrix();

  template <specfem::element::medium_tag MediumTag>
  void invert_mass_matrix(const type_real &dt);

  // template <specfem::element::medium_tag MediumTag,
  //           specfem::element::property_tag PropertyTag,
  //           specfem::element::boundary_tag BoundaryTag>
  // void compute_mass_matrix(const type_real &dt);
}; // namespace impl

// template <typename qp_type>
// class domain_kernels<qp_type, specfem::enums::simulation::type::adjoint> {
// public:
//   using elastic_type = specfem::enums::element::medium::elastic;
//   using acoustic_type = specfem::enums::element::medium::acoustic;
//   constexpr static auto forward_type =
//       specfem::enums::simulation::type::forward;
//   constexpr static auto adjoint_type =
//       specfem::enums::simulation::type::adjoint;
//   domain_kernels(const specfem::compute::assembly &assembly,
//                  const qp_type &quadrature_points);

//   void prepare_wavefields();

//   template <specfem::enums::element::medium medium>
//   void update_wavefields(const int istep);

// private:
//   specfem::kernels::impl::domain_kernels_impl<qp_type, forward_type>
//       forward_kernels;
//   specfem::kernels::impl::domain_kernels_impl<qp_type, adjoint_type>
//       adjoint_kernels;
// };
} // namespace impl
} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_IMPL_DOMAIN_KERNELS_HPP */
