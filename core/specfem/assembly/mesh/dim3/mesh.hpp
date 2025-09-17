#pragma once

#include "impl/control_nodes.hpp"
#include "impl/points.hpp"
#include "impl/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem::assembly {
/**
 * @brief Information on an assembled mesh
 *
 */
template <>
struct mesh<specfem::dimension::type::dim3>
    : public specfem::assembly::mesh_impl::points<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::quadrature<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::control_nodes<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim3> {

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension
  int nspec;                          ///< Number of spectral
                                      ///< elements
  int ngnod;                          ///< Number of control
                                      ///< nodes

  specfem::mesh_entity::element<dimension_tag> element_grid; ///< Element number
                                                             ///< of GLL points

  mesh() = default;

  mesh(const specfem::mesh::parameters<dimension_tag> &parameters,
       const specfem::mesh::coordinates<dimension_tag> &coordinates,
       const specfem::mesh::mapping<dimension_tag> &mapping,
       const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
       const specfem::quadrature::quadratures &quadrature);
};

} // namespace specfem::assembly
