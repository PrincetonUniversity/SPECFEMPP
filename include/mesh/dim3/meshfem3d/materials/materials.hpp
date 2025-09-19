#pragma once

namespace spectral::mesh::meshfem3d {

template <specfem::dimension::type Dimension> struct Materials {
  static constexpr auto dimension_tag = Dimension;

  struct material_specification {
    specfem::element::medium_tag type;
    specfem::element::property_tag property;
    int index;
    int database_index;

    material_specification() = default;

    material_specification(specfem::element::medium_tag type,
                           specfem::element::property_tag property, int index,
                           int database_index)
        : type(type), property(property), index(index),
          database_index(database_index) {};
  };

  template <specfem::element::medium_tag type,
            specfem::element::property_tag property>
  struct material {
    int n_materials;
    std::vector<specfem::medium::material<type, property> > element_materials;

    material() = default;

    material(const int n_materials,
             const std::vector<specfem::medium::material<type, property> >
                 &l_material);
  };

  int n_materials;
  specfem::kokkos::HostView1d<material_specification> material_index_mapping;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ACOUSTIC, ELASTIC),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
                      DECLARE(((specfem::mesh::materials, (_DIMENSION_TAG_),
                                ::material, (_MEDIUM_TAG_, _PROPERTY_TAG_)),
                               material)))

  materials() = default;

  materials(const int nspec, const int numat)
      : n_materials(numat),
        material_index_mapping("specfem::mesh::material_index_mapping", nspec) {
        };

public:
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  specfem::medium::material<MediumTag, PropertyTag>
  get_material(const int index) const {
    const auto &material_specification = this->material_index_mapping(index);

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ACOUSTIC, ELASTIC),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
        CAPTURE(material) {
          if constexpr (MediumTag == _medium_tag_ &&
                        PropertyTag == _property_tag_) {
            return _material_.element_materials[material_specification.index];
          }
        })

    Kokkos::abort("Invalid material type detected in material specification");

    return {};
  }

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  specfem::mesh::materials<dimension_tag>::material<MediumTag, PropertyTag> &
  get_container() {

    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ACOUSTIC, ELASTIC),
                         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
                        CAPTURE(material) {
                          if constexpr (_medium_tag_ == MediumTag &&
                                        _property_tag_ == PropertyTag) {
                            return _material_;
                          }
                        })

    Kokkos::abort("Invalid material type detected in material specification");
  }
};
} // namespace spectral::mesh::meshfem3d
