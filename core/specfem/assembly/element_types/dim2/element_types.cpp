#include "specfem/assembly/element_types.hpp"

specfem::assembly::element_types<specfem::dimension::type::dim2>::element_types(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : nspec(nspec),
      medium_tags("specfem::assembly::element_types::medium_tags", nspec),
      property_tags("specfem::assembly::element_types::property_tags", nspec),
      boundary_tags("specfem::assembly::element_types::boundary_tags", nspec) {

  for (int ispec = 0; ispec < nspec; ispec++) {
    const int ispec_mesh = mesh.compute_to_mesh(ispec);
    medium_tags(ispec) = tags.tags_container(ispec_mesh).medium_tag;
    property_tags(ispec) = tags.tags_container(ispec_mesh).property_tag;
    boundary_tags(ispec) = tags.tags_container(ispec_mesh).boundary_tag;
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(elements, h_elements) {
        int count = 0;
        int index = 0;
        for (int ispec = 0; ispec < nspec; ispec++) {
          if (medium_tags(ispec) == _medium_tag_) {
            count++;
          }
        }
        _elements_ =
            IndexViewType("specfem::assembly::element_types::elements", count);
        _h_elements_ = Kokkos::create_mirror_view(_elements_);
        for (int ispec = 0; ispec < nspec; ispec++) {
          if (medium_tags(ispec) == _medium_tag_) {
            _h_elements_(index) = ispec;
            index++;
          }
        }
        Kokkos::deep_copy(_elements_, _h_elements_);
      })

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
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
            IndexViewType("specfem::assembly::element_types::elements", count);
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

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
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
            IndexViewType("specfem::assembly::element_types::elements", count);
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
specfem::assembly::element_types<specfem::dimension::type::dim2>::
    get_elements_on_host(const specfem::element::medium_tag medium_tag) const {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(h_elements) {
        if (_medium_tag_ == medium_tag) {
          return _h_elements_;
        }
      })

  throw std::runtime_error("Medium tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace> specfem::assembly::
    element_types<specfem::dimension::type::dim2>::get_elements_on_device(
        const specfem::element::medium_tag medium_tag) const {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(elements) {
        if (_medium_tag_ == medium_tag) {
          return _elements_;
        }
      })

  throw std::runtime_error("Medium tag not found");
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> specfem::assembly::
    element_types<specfem::dimension::type::dim2>::get_elements_on_host(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(h_elements) {
        if (_medium_tag_ == medium_tag && _property_tag_ == property_tag) {
          return _h_elements_;
        }
      })

  throw std::runtime_error("Medium tag or property tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace> specfem::assembly::
    element_types<specfem::dimension::type::dim2>::get_elements_on_device(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(elements) {
        if (_medium_tag_ == medium_tag && _property_tag_ == property_tag) {
          return _elements_;
        }
      })

  throw std::runtime_error("Medium tag or property tag not found");
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> specfem::assembly::
    element_types<specfem::dimension::type::dim2>::get_elements_on_host(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag,
        const specfem::element::boundary_tag boundary_tag) const {
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      CAPTURE(h_elements) {
                        if (_medium_tag_ == medium_tag &&
                            _property_tag_ == property_tag &&
                            _boundary_tag_ == boundary_tag) {
                          return _h_elements_;
                        }
                      })

  throw std::runtime_error(
      "Medium tag, property tag or boundary tag not found");
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace> specfem::assembly::
    element_types<specfem::dimension::type::dim2>::get_elements_on_device(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag,
        const specfem::element::boundary_tag boundary_tag) const {

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      CAPTURE(elements) {
                        if (_medium_tag_ == medium_tag &&
                            _property_tag_ == property_tag &&
                            _boundary_tag_ == boundary_tag) {
                          return _elements_;
                        }
                      })

  throw std::runtime_error(
      "Medium tag, property tag or boundary tag not found");
}
