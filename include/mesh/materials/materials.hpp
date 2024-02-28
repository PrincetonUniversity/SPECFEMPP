#ifndef _MATERIAL_INDIC_HPP
#define _MATERIAL_INDIC_HPP

#include "kokkos_abstractions.h"
#include "material/material.hpp"
#include "specfem_mpi/interface.hpp"
#include <variant>

namespace specfem {
namespace mesh {
struct materials {

  struct material_specification {
    specfem::enums::element::type type;             ///< Type of element
    specfem::enums::element::property_tag property; ///< Property of element
    int index; ///< Index of material property

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
    material_specification(specfem::enums::element::type type,
                           specfem::enums::element::property_tag property,
                           int index)
        : type(type), property(property), index(index) {}
  };

  template <specfem::enums::element::type type,
            specfem::enums::element::property_tag property>
  struct material {
    int n_materials; ///< Number of elements
    specfem::kokkos::HostView1d<specfem::material::material<type, property> >
        material_properties; ///< Material properties

    material() = default;

    material(const int n_materials,
             const std::vector<specfem::material::material<type, property> >
                 &l_material);
  };

  int n_materials; ///< Total number of different materials
  specfem::kokkos::HostView1d<material_specification>
      material_index_mapping; ///< Mapping of spectral element to material
                              ///< properties

  specfem::mesh::materials::material<
      specfem::enums::element::type::elastic,
      specfem::enums::element::property_tag::isotropic>
      elastic_isotropic; ///< Elastic isotropic material properties

  specfem::mesh::materials::material<
      specfem::enums::element::type::acoustic,
      specfem::enums::element::property_tag::isotropic>
      acoustic_isotropic; ///< Acoustic isotropic material properties

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
  materials(const int nspec, const int ngnod);
  /**
   * @brief Constructor used to allocate and assign views from fortran database
   * file
   *
   * @param stream Stream object for fortran binary file buffered to material
   * definition section
   * @param ngnod Number of control nodes per spectral element
   * @param nspec Number of spectral elements
   * @param numat Total number of different materials
   * @param mpi Pointer to a MPI object
   */
  materials(std::ifstream &stream, const int numat, const int nspec,
            const specfem::kokkos::HostView2d<int> knods,
            const specfem::MPI::MPI *mpi);

  std::variant<specfem::material::material<
                   specfem::enums::element::type::elastic,
                   specfem::enums::element::property_tag::isotropic>,
               specfem::material::material<
                   specfem::enums::element::type::acoustic,
                   specfem::enums::element::property_tag::isotropic> >
  operator[](const int index) const;
};
} // namespace mesh
} // namespace specfem

#endif
