#pragma once

#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh/mesh_base.hpp"

#include "specfem_setup.hpp"
#include <variant>

namespace specfem {
namespace mesh {
/**
 * @brief Material properties information
 *
 */

template <> struct materials<specfem::dimension::type::dim2> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension type

  struct material_specification {
    specfem::element::medium_tag type;       ///< Type of element
    specfem::element::property_tag property; ///< Property of element
    int index;                               ///< Index of material property
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
                           specfem::element::property_tag property, int index,
                           int database_index)
        : type(type), property(property), index(index),
          database_index(database_index) {};
  };

  template <specfem::element::medium_tag type,
            specfem::element::property_tag property>
  struct material {
    int n_materials; ///< Number of elements
    std::vector<specfem::medium::material<dimension_tag, type, property> >
        element_materials; ///< Material properties

    material() = default;

    material(const int n_materials,
             const std::vector<
                 specfem::medium::material<dimension_tag, type, property> >
                 &l_material);
  };

  int n_materials; ///< Total number of different materials
  specfem::kokkos::HostView1d<material_specification>
      material_index_mapping; ///< Mapping of spectral element to material
                              ///< properties

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T,
                  ELECTROMAGNETIC_TE),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      DECLARE(((specfem::mesh::materials, (_DIMENSION_TAG_), ::material,
                (_MEDIUM_TAG_, _PROPERTY_TAG_)),
               material)))

  specfem::mesh::materials<specfem::dimension::type::dim2>::material<
      specfem::element::medium_tag::electromagnetic_te,
      specfem::element::property_tag::isotropic>
      electromagnetic_te_isotropic; ///< Electromagnetic material propertie TE

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
   * @param ngnod Number of control nodes per spectral element
   */
  materials(const int nspec, const int numat)
      : n_materials(numat),
        material_index_mapping("specfem::mesh::material_index_mapping", nspec) {
        };

  ///@}

public:
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  /**
   * @brief Material material at spectral element index
   *
   * @param index Spectral element index
   * @return std::variant Material properties
   */
  specfem::medium::material<dimension_tag, MediumTag, PropertyTag>
  get_material(const int index) const {
    const auto &material_specification = this->material_index_mapping(index);

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T, ELECTROMAGNETIC_TE),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        CAPTURE(material) {
          if constexpr (MediumTag == _medium_tag_ &&
                        PropertyTag == _property_tag_) {
            return _material_.element_materials[material_specification.index];
          }
        })

    Kokkos::abort("Invalid material type detected in material specification");

    return {};
  }

  /**
   * @brief Get the container object containing properties for a material type
   *
   * @tparam MediumTag Medium tag for the material
   * @tparam PropertyTag Property tag for the material
   * @return material<MediumTag, PropertyTag>& material container
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  specfem::mesh::materials<dimension_tag>::material<MediumTag, PropertyTag> &
  get_container() {

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T, ELECTROMAGNETIC_TE),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        CAPTURE(material) {
          if constexpr (_medium_tag_ == MediumTag &&
                        _property_tag_ == PropertyTag) {
            return _material_;
          }
        })

    Kokkos::abort("Invalid material type detected in material specification");
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
};

} // namespace mesh
} // namespace specfem
