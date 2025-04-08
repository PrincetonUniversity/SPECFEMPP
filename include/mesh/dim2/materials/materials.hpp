#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <variant>

namespace specfem {
namespace mesh {
/**
 * @brief Material properties information
 *
 */

template <> struct materials<specfem::dimension::type::dim2> {
  constexpr static auto dimension =
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
    std::vector<specfem::medium::material<type, property> >
        element_materials; ///< Material properties

    material() = default;

    material(const int n_materials,
             const std::vector<specfem::medium::material<type, property> >
                 &l_material);
  };

  int n_materials; ///< Total number of different materials
  specfem::kokkos::HostView1d<material_specification>
      material_index_mapping; ///< Mapping of spectral element to material
                              ///< properties

  FOR_EACH(IN_PRODUCT((DIMENSION_TAG_DIM2),
                      (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                       MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
                       MEDIUM_TAG_ELECTROMAGNETIC_TE),
                      (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
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

private:
#define TYPE_NAME(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                     \
  (CREATE_VARIABLE_NAME(type, GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)))

public:
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  /**
   * @brief Material material at spectral element index
   *
   * @param index Spectral element index
   * @return std::variant Material properties
   */
  specfem::medium::material<MediumTag, PropertyTag>
  get_material(const int index) const {
    const auto &material_specification = this->material_index_mapping(index);

    FOR_EACH(
        IN_PRODUCT((DIMENSION_TAG_DIM2),
                   (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                    MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
                    MEDIUM_TAG_ELECTROMAGNETIC_TE),
                   (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
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
  specfem::mesh::materials<dimension>::material<MediumTag, PropertyTag> &
  get_container() {

    FOR_EACH(
        IN_PRODUCT((DIMENSION_TAG_DIM2),
                   (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                    MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC,
                    MEDIUM_TAG_ELECTROMAGNETIC_TE),
                   (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
        CAPTURE(material) {
          if constexpr (_medium_tag_ == MediumTag &&
                        _property_tag_ == PropertyTag) {
            return _material_;
          }
        })

    Kokkos::abort("Invalid material type detected in material specification");
  }
};

} // namespace mesh
} // namespace specfem
