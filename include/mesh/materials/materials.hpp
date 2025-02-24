#pragma once

#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "specfem_mpi/interface.hpp"
#include <variant>

namespace specfem {
namespace mesh {
/**
 * @brief Material properties information
 *
 */
struct materials {

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
          database_index(database_index){};
  };

  template <specfem::element::medium_tag type,
            specfem::element::property_tag property>
  struct material {
    int n_materials; ///< Number of elements
    std::vector<specfem::medium::material<type, property> >
        material_properties; ///< Material properties

    material() = default;

    material(const int n_materials,
             const std::vector<specfem::medium::material<type, property> >
                 &l_material);
  };

  int n_materials; ///< Total number of different materials
  specfem::kokkos::HostView1d<material_specification>
      material_index_mapping; ///< Mapping of spectral element to material
                              ///< properties

  specfem::mesh::materials::material<specfem::element::medium_tag::elastic_sv,
                                     specfem::element::property_tag::isotropic>
      elastic_sv_isotropic; ///< Elastic isotropic material properties

  specfem::mesh::materials::material<
      specfem::element::medium_tag::elastic_sv,
      specfem::element::property_tag::anisotropic>
      elastic_sv_anisotropic; ///< Elastic anisotropic material properties

  specfem::mesh::materials::material<specfem::element::medium_tag::acoustic,
                                     specfem::element::property_tag::isotropic>
      acoustic_isotropic; ///< Acoustic isotropic material properties

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
        material_index_mapping("specfem::mesh::material_index_mapping",
                               nspec){};

  ///@}

  /**
   * @brief Overloaded operator to access material properties
   *
   * @param index Index of material properties
   * @return std::variant Material properties
   */
  std::variant<
      specfem::medium::material<specfem::element::medium_tag::elastic_sv,
                                specfem::element::property_tag::isotropic>,
      specfem::medium::material<specfem::element::medium_tag::elastic_sv,
                                specfem::element::property_tag::anisotropic>,
      specfem::medium::material<specfem::element::medium_tag::acoustic,
                                specfem::element::property_tag::isotropic> >
  operator[](const int index) const;
};
} // namespace mesh
} // namespace specfem
