#pragma once

#include "specfem/element.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/tag_dispatch/for_each.hpp"
#include "specfem/tag_dispatch/storage.hpp"
#include "specfem/tags.hpp"

#include "specfem/setup.hpp"
#include <variant>

namespace specfem {
namespace mesh {
/**
 * @brief Material properties information
 *
 */

template <> struct materials<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim2; ///< Dimension type

  struct material_specification {
    specfem::element::medium_tag type;             ///< Type of element
    specfem::element::property_tag property;       ///< Property of element
    specfem::element::attenuation_tag attenuation; ///< Attenuation type
    int index;          ///< Index of material property
    int database_index; ///< Index of material property in the database

    /**
     * @brief Default constructor
     *
     */
    material_specification() = default;

    /**
     * @brief Constructor used to assign values
     *
     * @param type Type of element
     * @param property Property of element
     * @param index Index of material property
     */
    material_specification(specfem::element::medium_tag type,
                           specfem::element::property_tag property,
                           specfem::element::attenuation_tag attenuation,
                           int index, int database_index)
        : type(type), property(property), attenuation(attenuation),
          index(index), database_index(database_index) {};
  };

  template <specfem::element::medium_tag type,
            specfem::element::property_tag property,
            specfem::element::attenuation_tag attenuation>
  struct material {
    int n_materials = 0; ///< Number of different materials
    std::vector<
        specfem::medium_container::material<dimension_tag, type, property,
                                            attenuation> >
        element_materials; ///< Material properties

    material() = default;

    material(const int n_materials,
             const std::vector<specfem::medium_container::material<
                 dimension_tag, type, property, attenuation> > &l_materials)
        : n_materials(n_materials), element_materials(l_materials) {}
  };

  int n_materials = 0; ///< Total number of different materials
  Kokkos::View<material_specification *, Kokkos::HostSpace>
      material_index_mapping; ///< Mapping of spectral element to material
                              ///< properties

  static constexpr auto combinations =
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
      specfem::tag_dispatch::medium_set<
          specfem::element::medium_tag::elastic_psv,
          specfem::element::medium_tag::elastic_sh,
          specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::poroelastic,
          specfem::element::medium_tag::elastic_psv_t,
          specfem::element::medium_tag::electromagnetic_te>{} *
      specfem::tag_dispatch::property_set<
          specfem::element::property_tag::isotropic,
          specfem::element::property_tag::anisotropic,
          specfem::element::property_tag::isotropic_cosserat>{} *
      specfem::tag_dispatch::attenuation_set<
          specfem::element::attenuation_tag::none,
          specfem::element::attenuation_tag::constant_isotropic>{};

  template <typename TagsType>
  using MaterialContainerTemplate =
      material<TagsType::medium_tag, TagsType::property_tag,
               TagsType::attenuation_tag>;

  specfem::tag_dispatch::TypedStorage<MaterialContainerTemplate,
                                      decltype(combinations)>
      material_containers;

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  materials() = default;
  /**
   * @brief Constructor used to allocate views
   *
   * @param nspec Number of spectral elements
   */
  materials(const int nspec)
      : n_materials(0), material_index_mapping(
                            "specfem::mesh::material_index_mapping", nspec) {};

  ///@}

public:
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::attenuation_tag AttenuationTag>
  int add_material(
      const specfem::medium_container::material<
          dimension_tag, MediumTag, PropertyTag, AttenuationTag> &material) {
    this->n_materials += 1;
    auto &material_container =
        this->get_container<MediumTag, PropertyTag, AttenuationTag>();
    material_container.element_materials.push_back(material);
    material_container.n_materials += 1;
    return material_container.n_materials - 1;
  }
  /**
   * @brief Material material at spectral element index
   *
   * @param index Spectral element index
   * @return std::variant Material properties
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::attenuation_tag AttenuationTag>
  specfem::medium_container::material<dimension_tag, MediumTag, PropertyTag,
                                      AttenuationTag>
  get_material(const int index) const {
    const auto &material_specification = this->material_index_mapping(index);

    using Key = specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag,
                                    AttenuationTag>;
    return material_containers.template get<Key>()
        .element_materials[material_specification.index];
  }

  /**
   * @brief Get the container object containing properties for a material type
   *
   * @tparam MediumTag Medium tag for the material
   * @tparam PropertyTag Property tag for the material
   * @return material<MediumTag, PropertyTag>& material container
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::attenuation_tag AttenuationTag>
  specfem::mesh::materials<dimension_tag>::material<MediumTag, PropertyTag,
                                                    AttenuationTag> &
  get_container() {

    using Key = specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag,
                                    AttenuationTag>;
    return material_containers.template get<Key>();
  }

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag,
            specfem::element::attenuation_tag AttenuationTag>
  const specfem::mesh::materials<dimension_tag>::material<
      MediumTag, PropertyTag, AttenuationTag> &
  get_container() const {

    using Key = specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag,
                                    AttenuationTag>;
    return material_containers.template get<Key>();
  }

  /**
   * @brief Get the medium and property types for a given element index
   *
   * Returns a tuple containing the medium and property tags associated with
   * the material specification for the specified element index.
   *
   * @param index Element index to query
   * @return Tuple of (medium_tag, property_tag) for the element's material
   *
   * @code
   * // Get material type for element 42
   * auto [medium, property] = materials.get_material_type(42);
   *
   * // Use in conditional logic
   * if (medium == specfem::element::medium_tag::elastic &&
   *     property == specfem::element::property_tag::isotropic) {
   *     // Handle elastic isotropic case
   * }
   * @endcode
   */
  std::tuple<specfem::element::medium_tag, specfem::element::property_tag>
  get_material_type(const int index) const {
#ifndef NDEBUG
    if (index < 0 || index >= this->material_index_mapping.size()) {
      KOKKOS_ABORT_WITH_LOCATION(
          "Element index out of range in get_material_type");
    }
#endif
    const auto &material_specification = this->material_index_mapping[index];
    return std::make_tuple(material_specification.type,
                           material_specification.property);
  }

  void print() const;
};

} // namespace mesh
} // namespace specfem
