#include "compute/element_types/element_types.hpp"

specfem::compute::element_types::element_types(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : nspec(nspec),
      medium_tags("specfem::compute::element_types::medium_tags", nspec),
      property_tags("specfem::compute::element_types::property_tags", nspec),
      boundary_tags("specfem::compute::element_types::boundary_tags", nspec) {

  for (int ispec = 0; ispec < nspec; ispec++) {
    const int ispec_mesh = mapping.compute_to_mesh(ispec);
    medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
    property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
    boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
  }

#define COUNT_MEDIUM_TAG_INDICES(DIMENSION_TAG, MEDIUM_TAG)                    \
  int CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG)) = 0;                          \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (medium_tags(ispec) == GET_TAG(MEDIUM_TAG)) {                           \
      CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG))                               \
      ++;                                                                      \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      COUNT_MEDIUM_TAG_INDICES, WHERE(DIMENSION_TAG_DIM2) WHERE(
                                    MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef COUNT_MEDIUM_TAG_INDICES

#define ALLOCATE_MEDIUM_TAG_VIEWS(DIMENSION_TAG, MEDIUM_TAG)                   \
  this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),                \
                             GET_NAME(MEDIUM_TAG)) =                           \
      IndexViewType("specfem::compute::element_types::elements",               \
                    CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),       \
                                         GET_NAME(MEDIUM_TAG)));               \
  this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),              \
                             GET_NAME(MEDIUM_TAG)) =                           \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG)));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      ALLOCATE_MEDIUM_TAG_VIEWS,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef ALLOCATE_MEDIUM_TAG_VIEWS

#define ASSIGN_MEDIUM_TAG_INDICES(DIMENSION_TAG, MEDIUM_TAG)                   \
  int CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG)) = 0;                          \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (medium_tags(ispec) == GET_TAG(MEDIUM_TAG)) {                           \
      this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),          \
                                 GET_NAME(MEDIUM_TAG))(CREATE_VARIABLE_NAME(   \
          index, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG))) = ispec;      \
      CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG))                               \
      ++;                                                                      \
    }                                                                          \
  }                                                                            \
  Kokkos::deep_copy(                                                           \
      this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),            \
                                 GET_NAME(MEDIUM_TAG)),                        \
      this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),          \
                                 GET_NAME(MEDIUM_TAG)));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      ASSIGN_MEDIUM_TAG_INDICES,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef ASSIGN_MEDIUM_TAG_INDICES

#define COUNT_MATERIAL_SYSTEM_INDICES(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG) \
  int CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) = 0;  \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (medium_tags(ispec) == GET_TAG(MEDIUM_TAG) &&                           \
        property_tags(ispec) == GET_TAG(PROPERTY_TAG)) {                       \
      CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))       \
      ++;                                                                      \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      COUNT_MATERIAL_SYSTEM_INDICES,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef COUNT_MATERIAL_SYSTEM_INDICES

#define ALLOCATE_MATERIAL_SYSTEM_VIEWS(DIMENSION_TAG, MEDIUM_TAG,              \
                                       PROPERTY_TAG)                           \
  this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),                \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) =   \
      IndexViewType("specfem::compute::element_types::elements",               \
                    CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),       \
                                         GET_NAME(MEDIUM_TAG),                 \
                                         GET_NAME(PROPERTY_TAG)));             \
  this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),              \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) =   \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),             \
          GET_NAME(PROPERTY_TAG)));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      ALLOCATE_MATERIAL_SYSTEM_VIEWS,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef ALLOCATE_MATERIAL_SYSTEM_VIEWS

#define ASSIGN_MATERIAL_SYSTEM_INDICES(DIMENSION_TAG, MEDIUM_TAG,              \
                                       PROPERTY_TAG)                           \
  int CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) = 0;  \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (medium_tags(ispec) == GET_TAG(MEDIUM_TAG) &&                           \
        property_tags(ispec) == GET_TAG(PROPERTY_TAG)) {                       \
      this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),          \
                                 GET_NAME(MEDIUM_TAG),                         \
                                 GET_NAME(PROPERTY_TAG))(CREATE_VARIABLE_NAME( \
          index, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),                \
          GET_NAME(PROPERTY_TAG))) = ispec;                                    \
      CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))       \
      ++;                                                                      \
    }                                                                          \
  }                                                                            \
  Kokkos::deep_copy(this->CREATE_VARIABLE_NAME(                                \
                        elements, GET_NAME(DIMENSION_TAG),                     \
                        GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)),         \
                    this->CREATE_VARIABLE_NAME(                                \
                        h_elements, GET_NAME(DIMENSION_TAG),                   \
                        GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      ASSIGN_MATERIAL_SYSTEM_INDICES,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef ASSIGN_MATERIAL_SYSTEM_INDICES

#define COUNT_ELEMENT_TYPES_INDICES(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,   \
                                    BOUNDARY_TAG)                              \
  int CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (medium_tags(ispec) == GET_TAG(MEDIUM_TAG) &&                           \
        property_tags(ispec) == GET_TAG(PROPERTY_TAG) &&                       \
        boundary_tags(ispec) == GET_TAG(BOUNDARY_TAG)) {                       \
      CREATE_VARIABLE_NAME(count, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG))                             \
      ++;                                                                      \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      COUNT_ELEMENT_TYPES_INDICES,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
                  BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef COUNT_ELEMENT_TYPES_INDICES

#define ALLOCATE_ELEMENT_TYPES_VIEWS(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,  \
                                     BOUNDARY_TAG)                             \
  this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),                \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType("specfem::compute::element_types::elements",               \
                    CREATE_VARIABLE_NAME(                                      \
                        count, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),  \
                        GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)));      \
  this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),              \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),             \
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ALLOCATE_ELEMENT_TYPES_VIEWS,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
                  BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ALLOCATE_ELEMENT_TYPES_VIEWS

#define ASSIGN_ELEMENT_TYPES_INDICES(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,  \
                                     BOUNDARY_TAG)                             \
  int CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (medium_tags(ispec) == GET_TAG(MEDIUM_TAG) &&                           \
        property_tags(ispec) == GET_TAG(PROPERTY_TAG) &&                       \
        boundary_tags(ispec) == GET_TAG(BOUNDARY_TAG)) {                       \
      this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),          \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME( \
          index, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),                \
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = ispec;            \
      CREATE_VARIABLE_NAME(index, GET_NAME(DIMENSION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG))                             \
      ++;                                                                      \
    }                                                                          \
  }                                                                            \
  Kokkos::deep_copy(                                                           \
      this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),            \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)),                      \
      this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),          \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ASSIGN_ELEMENT_TYPES_INDICES,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
                  BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ASSIGN_ELEMENT_TYPES_INDICES
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag) const {

#define RETURN_VARIABLE(DIMENSION_TAG, MEDIUM_TAG)                             \
  if (GET_TAG(MEDIUM_TAG) == medium_tag) {                                     \
    return this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),     \
                                      GET_NAME(MEDIUM_TAG));                   \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      RETURN_VARIABLE, WHERE(DIMENSION_TAG_DIM2)
                           WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef RETURN_VARIABLE
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag) const {

#define RETURN_VARIABLE(DIMENSION_TAG, MEDIUM_TAG)                             \
  if (GET_TAG(MEDIUM_TAG) == medium_tag) {                                     \
    return this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),       \
                                      GET_NAME(MEDIUM_TAG));                   \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      RETURN_VARIABLE, WHERE(DIMENSION_TAG_DIM2)
                           WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC))

#undef RETURN_VARIABLE
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

#define RETURN_VARIABLE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)               \
  if (GET_TAG(MEDIUM_TAG) == medium_tag &&                                     \
      GET_TAG(PROPERTY_TAG) == property_tag) {                                 \
    return this->CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENSION_TAG),     \
                                      GET_NAME(MEDIUM_TAG),                    \
                                      GET_NAME(PROPERTY_TAG));                 \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      RETURN_VARIABLE,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef RETURN_VARIABLE
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

#define RETURN_VARIABLE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)               \
  if (GET_TAG(MEDIUM_TAG) == medium_tag &&                                     \
      GET_TAG(PROPERTY_TAG) == property_tag) {                                 \
    return this->CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),       \
                                      GET_NAME(MEDIUM_TAG),                    \
                                      GET_NAME(PROPERTY_TAG));                 \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      RETURN_VARIABLE,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef RETURN_VARIABLE
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag,
    const specfem::element::boundary_tag boundary_tag) const {

#define RETURN_VARIABLE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG) \
  if (GET_TAG(MEDIUM_TAG) == medium_tag &&                                     \
      GET_TAG(PROPERTY_TAG) == property_tag &&                                 \
      GET_TAG(BOUNDARY_TAG) == boundary_tag) {                                 \
    return this->CREATE_VARIABLE_NAME(                                         \
        h_elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),             \
        GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG));                       \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      RETURN_VARIABLE,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
                  BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef RETURN_VARIABLE
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag,
    const specfem::element::boundary_tag boundary_tag) const {

#define RETURN_VARIABLE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG) \
  if (GET_TAG(MEDIUM_TAG) == medium_tag &&                                     \
      GET_TAG(PROPERTY_TAG) == property_tag &&                                 \
      GET_TAG(BOUNDARY_TAG) == boundary_tag) {                                 \
    return this->CREATE_VARIABLE_NAME(                                         \
        elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),               \
        GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG));                       \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      RETURN_VARIABLE,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
                  BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef RETURN_VARIABLE
}
