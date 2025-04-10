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
      COUNT_MEDIUM_TAG_INDICES,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC))

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
          WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC))

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
          WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC))

#undef ASSIGN_MEDIUM_TAG_INDICES

  FOR_EACH(
      IN_PRODUCT((DIMENSION_TAG_DIM2),
                 (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                 (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
      CAPTURE(elements, h_elements) {
        int count = 0;
        int index = 0;

        for (int ispec = 0; ispec < nspec; ispec++) {
          if (medium_tags(ispec) == _medium_tag_ &&
              property_tags(ispec) == _property_tag_) {
            count++;
          }
        }

        _elements_ =
            IndexViewType("specfem::compute::element_types::elements", count);
        _h_elements_ = Kokkos::create_mirror_view(_elements_);

        for (int ispec = 0; ispec < nspec; ispec++) {
          if (medium_tags(ispec) == _medium_tag_ &&
              property_tags(ispec) == _property_tag_) {
            _h_elements_(index) = ispec;
            index++;
          }
        }

        Kokkos::deep_copy(_elements_, _h_elements_);
      })

  FOR_EACH(
      IN_PRODUCT((DIMENSION_TAG_DIM2),
                 (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                 (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
                 (BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY,
                  BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(elements, h_elements) {
        int count = 0;
        int index = 0;

        for (int ispec = 0; ispec < nspec; ispec++) {
          if (medium_tags(ispec) == _medium_tag_ &&
              property_tags(ispec) == _property_tag_ &&
              boundary_tags(ispec) == _boundary_tag_) {
            count++;
          }
        }

        _elements_ =
            IndexViewType("specfem::compute::element_types::elements", count);
        _h_elements_ = Kokkos::create_mirror_view(_elements_);

        for (int ispec = 0; ispec < nspec; ispec++) {
          if (medium_tags(ispec) == _medium_tag_ &&
              property_tags(ispec) == _property_tag_ &&
              boundary_tags(ispec) == _boundary_tag_) {
            _h_elements_(index) = ispec;
            index++;
          }
        }

        Kokkos::deep_copy(_elements_, _h_elements_);
      })
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
                           WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                                 MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC))

#undef RETURN_VARIABLE

  throw std::runtime_error("Medium tag not found");
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
                           WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                                 MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC))

#undef RETURN_VARIABLE

  throw std::runtime_error("Medium tag not found");
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

  FOR_EACH(
      IN_PRODUCT((DIMENSION_TAG_DIM2),
                 (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                 (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
      CAPTURE(h_elements) {
        if (_medium_tag_ == medium_tag && _property_tag_ == property_tag) {
          return _h_elements_;
        }
      })

  throw std::runtime_error("Medium tag or property tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

  FOR_EACH(
      IN_PRODUCT((DIMENSION_TAG_DIM2),
                 (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                 (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
      CAPTURE(elements) {
        if (_medium_tag_ == medium_tag && _property_tag_ == property_tag) {
          return _elements_;
        }
      })

  throw std::runtime_error("Medium tag or property tag not found");
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::element_types::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag,
    const specfem::element::boundary_tag boundary_tag) const {
  FOR_EACH(
      IN_PRODUCT((DIMENSION_TAG_DIM2),
                 (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                 (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
                 (BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY,
                  BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(h_elements) {
        if (_medium_tag_ == medium_tag && _property_tag_ == property_tag &&
            _boundary_tag_ == boundary_tag) {
          return _h_elements_;
        }
      })

  throw std::runtime_error(
      "Medium tag, property tag or boundary tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::element_types::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag,
    const specfem::element::boundary_tag boundary_tag) const {

  FOR_EACH(
      IN_PRODUCT((DIMENSION_TAG_DIM2),
                 (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC),
                 (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
                 (BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                  BOUNDARY_TAG_STACEY,
                  BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(elements) {
        if (_medium_tag_ == medium_tag && _property_tag_ == property_tag &&
            _boundary_tag_ == boundary_tag) {
          return _elements_;
        }
      })

  throw std::runtime_error(
      "Medium tag, property tag or boundary tag not found");
}
