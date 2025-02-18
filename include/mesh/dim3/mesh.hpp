#pragma once

#include "coordinates/coordinates.hpp"
#include "mesh/dim3/mapping/mapping.hpp"
#include "mesh/mesh_base.hpp"
#include "parameters/parameters.hpp"
#include "partial_derivatives/partial_derivatives.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {

namespace mesh {

template <> struct mesh<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  // Struct to store all the mesh parameter
  specfem::mesh::parameters<dimension> parameters;

  // Struct to store all the coordinates
  specfem::mesh::coordinates<dimension> coordinates;

  // Struct to store the mapping information
  specfem::mesh::mapping<dimension> mapping;

  // Irregular elements Kokkos
  type_real xix_regular, jacobian_regular;
  View1D<int> irregular_element_number;

  // Struct to store the partial derivatives
  specfem::mesh::partial_derivatives<dimension> partial_derivatives;

  //
  // int npgeo; ///< Total number of spectral element control nodes
  // int nspec; ///< Total number of spectral elements
  // int nproc; ///< Total number of processors
  // specfem::mesh::control_nodes<dimension> control_nodes; ///< Defines control
  //                                                        ///< nodes

  // specfem::mesh::parameters<dimension> parameters; ///< Struct to store
  //                                                  ///< simulation launch
  //                                                  ///< parameters (never
  //                                                  used)

  // specfem::mesh::coupled_interfaces<dimension>
  //     coupled_interfaces; ///< Struct to store
  //                         ///< coupled interfaces

  // specfem::mesh::boundaries<dimension> boundaries; ///< Struct to store
  //                                                  ///< information at the
  //                                                  ///< boundaries

  // specfem::mesh::tags<dimension> tags; ///< Struct to store
  //                                      ///< tags for every
  //                                      ///< spectral
  //                                      ///< element

  // specfem::mesh::elements::tangential_elements<dimension>
  //     tangential_nodes; ///< Defines
  //                       ///< tangential
  //                       ///< nodes
  //                       ///< (never
  //                       ///< used)

  // specfem::mesh::elements::axial_elements<dimension> axial_nodes; ///<
  // Defines
  //                                                                 ///< axial
  //                                                                 ///< nodes
  //                                                                 ///< (never
  //                                                                 ///< used)
  // specfem::mesh::materials materials; ///< Defines material properties

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default mesh constructor
   *
   */
  mesh(){};

  // mesh(const int npgeo, const int nspec, const int nproc,
  //      const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
  //          &control_nodes,
  //      const specfem::mesh::parameters<specfem::dimension::type::dim2>
  //          &parameters,
  //      const
  //      specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>
  //          &coupled_interfaces,
  //      const specfem::mesh::boundaries<specfem::dimension::type::dim2>
  //          &boundaries,
  //      const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
  //      const specfem::mesh::elements::tangential_elements<
  //          specfem::dimension::type::dim2> &tangential_nodes,
  //      const specfem::mesh::elements::axial_elements<
  //          specfem::dimension::type::dim2> &axial_nodes,
  //      const specfem::mesh::materials &materials)
  //     : npgeo(npgeo), nspec(nspec), nproc(nproc),
  //     control_nodes(control_nodes),
  //       parameters(parameters), coupled_interfaces(coupled_interfaces),
  //       boundaries(boundaries), tags(tags),
  //       tangential_nodes(tangential_nodes), axial_nodes(axial_nodes),
  //       materials(materials){};
  ///@}

  std::string print() const;
};
} // namespace mesh
} // namespace specfem
