#ifndef _COMPUTE_HPP
#define _COMPUTE_HPP

// #include "compute/compute_quadrature.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "point/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

struct shape_functions {
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension
  int ngnod; ///< Number of control nodes
  specfem::kokkos::DeviceView3d<type_real> shape2D;    ///< Shape functions
  specfem::kokkos::DeviceView4d<type_real> dshape2D;   ///< Shape function
                                                       ///< derivatives
  specfem::kokkos::HostMirror3d<type_real> h_shape2D;  ///< Shape functions
  specfem::kokkos::HostMirror4d<type_real> h_dshape2D; ///< Shape function
                                                       ///< derivatives

  shape_functions(const int &ngllz, const int &ngllx, const int &ngnod)
      : ngllz(ngllz), ngllx(ngllx), ngnod(ngnod),
        shape2D("specfem::compute::shape_functions::shape2D", ngllz, ngllx,
                ngnod),
        dshape2D("specfem::compute::shape_functions::dshape2D", ngllz, ngllx,
                 ndim, ngnod),
        h_shape2D(Kokkos::create_mirror_view(shape2D)),
        h_dshape2D(Kokkos::create_mirror_view(dshape2D)) {}

  shape_functions(const specfem::kokkos::HostMirror1d<type_real> xi,
                  const specfem::kokkos::HostMirror1d<type_real> gamma,
                  const int &ngll, const int &ngnod);
};

struct quadrature {
  struct GLL {
    int N; ///< Number of quadrature points
    specfem::kokkos::DeviceView1d<type_real> xi;       ///< Quadrature points
    specfem::kokkos::HostMirror1d<type_real> h_xi;     ///< Quadrature points
    specfem::compute::shape_functions shape_functions; ///< Shape functions

    GLL(const specfem::quadrature::quadratures &quadratures, const int &ngnod)
        : N(quadratures.gll.get_N()), xi(quadratures.gll.get_xi()),
          h_xi(quadratures.gll.get_hxi()), shape_functions(xi, xi, N, ngnod) {}
  };

  specfem::compute::quadrature::GLL gll; ///< GLL object

  quadrature(const specfem::quadrature::quadratures &quadratures,
             const specfem::mesh::control_nodes &control_nodes)
      : gll(quadratures, control_nodes.ngnod) {}
};

struct control_nodes {
  int nspec; ///< Number of spectral elements
  int ngnod; ///< Number of control nodes
  specfem::kokkos::DeviceView2d<int> index_mapping; ///< Global delement
                                                    ///< number for every
                                                    ///< control node
  specfem::kokkos::DeviceView3d<type_real> coord; ///< (x, z) for every distinct
                                                  ///< control node
  specfem::kokkos::HostMirror2d<int> h_index_mapping; ///< Global control
                                                      ///< element number for
                                                      ///< every control node
  specfem::kokkos::HostMirror3d<type_real> h_coord;   ///< (x, z) for every
                                                      ///< distinct control node

  control_nodes(const specfem::mesh::control_nodes &control_nodes);
};

struct points {
  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension
  specfem::kokkos::DeviceView3d<int> index_mapping; ///< Global element
                                                    ///< number for every
                                                    ///< quadrature point
  specfem::kokkos::DeviceView4d<type_real> coord; ///< (x, z) for every distinct
                                                  ///< quadrature point
  specfem::kokkos::HostMirror3d<int> h_index_mapping; ///< Global element
                                                      ///< number for every
                                                      ///< quadrature point
  specfem::kokkos::HostMirror4d<type_real> h_coord;   ///< (x, z) for every
                                                      ///< distinct quadrature
                                                      ///< point
  type_real xmin, xmax, zmin, zmax; ///< Min and max values of x and z
                                    ///< coordinates

  points() = default;

  points(const int &nspec, const int &ngllz, const int &ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        index_mapping("specfem::compute::points::index_mapping", nspec, ngllz,
                      ngllx),
        coord("specfem::compute::points::coord", ndim, nspec, ngllz, ngllx),
        h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
        h_coord(Kokkos::create_mirror_view(coord)) {}
};

/**
 * @brief Information on an assembled mesh
 *
 */
struct mesh {
  specfem::compute::control_nodes control_nodes; ///< Control nodes
  specfem::compute::points points;               ///< Quadrature points
  specfem::compute::quadrature quadratures;      ///< Quadrature object

  mesh(const specfem::mesh::control_nodes &control_nodes,
       const specfem::quadrature::quadratures &quadratures);

  specfem::compute::points assemble();

  specfem::point::gcoord2 locate(const specfem::point::lcoord2 &point);

  specfem::point::lcoord2 locate(const specfem::point::gcoord2 &point);
};

} // namespace compute
} // namespace specfem

#endif
