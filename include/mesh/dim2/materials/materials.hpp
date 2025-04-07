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

#define DEFINE_MATERIAL_CONTAINER(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)     \
  specfem::mesh::materials<GET_TAG(                                            \
      DIMENSION_TAG)>::material<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>    \
      CREATE_VARIABLE_NAME(material, GET_NAME(MEDIUM_TAG),                     \
                           GET_NAME(PROPERTY_TAG));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      DEFINE_MATERIAL_CONTAINER,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ELASTIC_PSV_T, MEDIUM_TAG_ACOUSTIC,
                MEDIUM_TAG_POROELASTIC, MEDIUM_TAG_ELECTROMAGNETIC_TE)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC,
                    PROPERTY_TAG_ISOTROPIC_COSSERAT))

#undef DEFINE_MATERIAL_CONTAINER
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
#define SOURCE_MEDIUM_STORE_ON_DEVICE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG) \
  using CREATE_VARIABLE_NAME(type, GET_NAME(MEDIUM_TAG),                       \
                             GET_NAME(PROPERTY_TAG)) =                         \
      specfem::medium::material<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>;

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      SOURCE_MEDIUM_STORE_ON_DEVICE,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ELASTIC_PSV_T, MEDIUM_TAG_ACOUSTIC,
                MEDIUM_TAG_POROELASTIC, MEDIUM_TAG_ELECTROMAGNETIC_TE)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC,
                    PROPERTY_TAG_ISOTROPIC_COSSERAT))

#undef SOURCE_MEDIUM_STORE_ON_DEVICE

#define TYPE_NAME(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                     \
  (CREATE_VARIABLE_NAME(type, GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)))

public:
#define MAKE_VARIANT_RETURN
  std::variant<BOOST_PP_SEQ_ENUM(CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TYPE_NAME, WHERE(DIMENSION_TAG_DIM2) WHERE(
                     MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                     MEDIUM_TAG_ELASTIC_PSV_T, MEDIUM_TAG_ACOUSTIC,
                     MEDIUM_TAG_POROELASTIC, MEDIUM_TAG_ELECTROMAGNETIC_TE)
                     WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC,
                           PROPERTY_TAG_ISOTROPIC_COSSERAT)))>

      /**
       * @brief Material material at spectral element index
       *
       * @param index Spectral element index
       * @return std::variant Material properties
       */
      MAKE_VARIANT_RETURN operator[](const int index) const {

#undef MAKE_VARIANT_RETURN
#undef TYPE_NAME

    const auto &material_specification = this->material_index_mapping(index);

#define RETURN_VALUE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                  \
  if (material_specification.type == GET_TAG(MEDIUM_TAG) &&                    \
      material_specification.property == GET_TAG(PROPERTY_TAG)) {              \
    return this                                                                \
        ->CREATE_VARIABLE_NAME(material, GET_NAME(MEDIUM_TAG),                 \
                               GET_NAME(PROPERTY_TAG))                         \
        .element_materials[material_specification.index];                      \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        RETURN_VALUE,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ELASTIC_PSV_T, MEDIUM_TAG_ACOUSTIC,
                  MEDIUM_TAG_POROELASTIC, MEDIUM_TAG_ELECTROMAGNETIC_TE)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC,
                      PROPERTY_TAG_ISOTROPIC_COSSERAT))

#undef RETURN_VALUE

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

#define RETURN_VALUE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                  \
  if constexpr (GET_TAG(MEDIUM_TAG) == MediumTag &&                            \
                GET_TAG(PROPERTY_TAG) == PropertyTag) {                        \
    return this->CREATE_VARIABLE_NAME(material, GET_NAME(MEDIUM_TAG),          \
                                      GET_NAME(PROPERTY_TAG));                 \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        RETURN_VALUE,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ELASTIC_PSV_T, MEDIUM_TAG_ACOUSTIC,
                  MEDIUM_TAG_POROELASTIC, MEDIUM_TAG_ELECTROMAGNETIC_TE)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC,
                      PROPERTY_TAG_ISOTROPIC_COSSERAT))

#undef RETURN_VALUE

    Kokkos::abort("Invalid material type detected in material specification");
  }

  // #undef MAKE_VARIANT_RETURN
};

} // namespace mesh
} // namespace specfem
