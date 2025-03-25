#include "compute/kernels/kernels.hpp"

specfem::compute::kernels::kernels(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::element_types &element_types) {

  this->nspec = nspec;
  this->ngllz = ngllz;
  this->ngllx = ngllx;

  this->property_index_mapping =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
          "specfem::compute::kernels::property_index_mapping", nspec);

  this->h_property_index_mapping =
      Kokkos::create_mirror_view(property_index_mapping);

#define GET_ELEMENTS(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                  \
  const auto CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),           \
                                  GET_NAME(MEDIUM_TAG),                        \
                                  GET_NAME(PROPERTY_TAG)) =                    \
      element_types.get_elements_on_host(GET_TAG(MEDIUM_TAG),                  \
                                         GET_TAG(PROPERTY_TAG));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      GET_ELEMENTS,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC));

#undef GET_ELEMENTS

  for (int ispec = 0; ispec < nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

#define ASSIGN_KERNEL_CONTAINERS(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)      \
  CREATE_VARIABLE_NAME(value, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),   \
                       GET_NAME(PROPERTY_TAG)) =                               \
      specfem::medium::kernels_container<GET_TAG(MEDIUM_TAG),                  \
                                         GET_TAG(PROPERTY_TAG)>(               \
          CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),              \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)),  \
          ngllz, ngllx, h_property_index_mapping);

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      ASSIGN_KERNEL_CONTAINERS,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC));

#undef ASSIGN_KERNEL_CONTAINERS

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}
