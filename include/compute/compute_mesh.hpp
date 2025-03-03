#pragma once

// #include "compute/compute_quadrature.hpp"
#include "element/quadrature.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "point/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Mapping between spectral element indexing within @ref
 * specfem::mesh::mesh and @ref specfem::compute::mesh
 *
 * We reorder the mesh to enable better memory access patterns when computing
 * forces.
 *
 */
struct mesh_to_compute_mapping {
  int nspec; ///< Number of spectral elements
  specfem::kokkos::HostView1d<int> compute_to_mesh; ///< Mapping from compute
                                                    ///< ordering to mesh
                                                    ///< ordering
  specfem::kokkos::HostView1d<int> mesh_to_compute; ///< Mapping from mesh
                                                    ///< ordering to compute
                                                    ///< ordering

  mesh_to_compute_mapping() = default;

  mesh_to_compute_mapping(
      const specfem::mesh::tags<specfem::dimension::type::dim2> &tags);
};

/**
 * @brief Shape function and their derivatives for every control node within the
 * mesh
 *
 */
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

  shape_functions() = default;
};

/**
 * @brief Information about the integration quadratures
 *
 */
struct quadrature {
  struct GLL {
    int N; ///< Number of quadrature points
    specfem::kokkos::DeviceView1d<type_real> xi;        ///< Quadrature points
    specfem::kokkos::HostMirror1d<type_real> h_xi;      ///< Quadrature points
    specfem::compute::shape_functions shape_functions;  ///< Shape functions
    specfem::kokkos::DeviceView1d<type_real> weights;   ///< Quadrature weights
    specfem::kokkos::HostMirror1d<type_real> h_weights; ///< Quadrature weights
    specfem::kokkos::DeviceView2d<type_real> hprime;    ///< Derivative of
                                                     ///< lagrange interpolants
    specfem::kokkos::HostMirror2d<type_real> h_hprime; ///< Derivative of
                                                       ///< lagrange
                                                       ///< interpolants

    GLL() = default;

    GLL(const specfem::quadrature::quadratures &quadratures, const int &ngnod)
        : N(quadratures.gll.get_N()), xi(quadratures.gll.get_xi()),
          weights(quadratures.gll.get_w()), h_xi(quadratures.gll.get_hxi()),
          h_weights(quadratures.gll.get_hw()),
          hprime(quadratures.gll.get_hprime()),
          h_hprime(quadratures.gll.get_hhprime()),
          shape_functions(h_xi, h_xi, N, ngnod) {}
  };

  specfem::compute::quadrature::GLL gll; ///< GLL quadrature

  quadrature() = default;

  quadrature(const specfem::quadrature::quadratures &quadratures,
             const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
                 &control_nodes)
      : gll(quadratures, control_nodes.ngnod) {}
};

/**
 * @brief Spectral element control nodes
 *
 */
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

  control_nodes(
      const specfem::compute::mesh_to_compute_mapping &mapping,
      const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
          &control_nodes);

  control_nodes() = default;
};

/**
 * @brief Spectral element assembly information
 *
 */
struct points {
  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension

  using ViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  ViewType index_mapping;                         ///< Global index
                                                  ///< number for every
                                                  ///< quadrature point
  specfem::kokkos::DeviceView4d<type_real> coord; ///< (x, z) for every distinct
                                                  ///< quadrature point
  ViewType::HostMirror h_index_mapping;           ///< Global element
                                                  ///< number for every
                                                  ///< quadrature point
  specfem::kokkos::HostMirror4d<type_real> h_coord; ///< (x, z) for every
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
  int nspec;                                         ///< Number of spectral
                                                     ///< elements
  int ngllz;                                         ///< Number of quadrature
                                                     ///< points in z dimension
  int ngllx;                                         ///< Number of quadrature
                                                     ///< points in x dimension
  specfem::compute::control_nodes control_nodes;     ///< Control nodes
  specfem::compute::points points;                   ///< Quadrature points
  specfem::compute::quadrature quadratures;          ///< Quadrature object
  specfem::compute::mesh_to_compute_mapping mapping; ///< Mapping of spectral
                                                     ///< element index between
                                                     ///< mesh database ordering
                                                     ///< and compute ordering

  mesh() = default;

  mesh(const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
       const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
           &control_nodes,
       const specfem::quadrature::quadratures &quadratures);

  specfem::compute::points assemble();

  /**
   * @brief Compute the global coordinates for a point given its local
   * coordinates
   *
   * @param point Local coordinates
   * @return specfem::point::global_coordinates<specfem::dimension::type::dim2>
   * Global coordinates
   */
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
  locate(const specfem::point::local_coordinates<specfem::dimension::type::dim2>
             &point);

  /**
   * @brief Compute the local coordinates for a point given its global
   * coordinates
   *
   * @param point Global coordinates
   * @return specfem::point::local_coordinates<specfem::dimension::type::dim2>
   * Local coordinates
   */
  specfem::point::local_coordinates<specfem::dimension::type::dim2> locate(
      const specfem::point::global_coordinates<specfem::dimension::type::dim2>
          &point);
};

/**
 * @defgroup QuadratureDataAccess
 *
 */

/**
 * @brief Load quadrature data for a spectral element on host or device
 *
 * @ingroup QuadratureDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref specfem::element::quadrature
 * @param team Team member
 * @param quadrature Quadrature data
 * @param element_quadrature Quadrature data for the element (output)
 */
template <bool on_device, typename MemberType, typename ViewType>
KOKKOS_INLINE_FUNCTION void
impl_load(const MemberType &team,
          const specfem::compute::quadrature &quadrature,
          ViewType &element_quadrature) {

  constexpr bool store_hprime_gll = ViewType::store_hprime_gll;

  constexpr bool store_weight_times_hprime_gll =
      ViewType::store_weight_times_hprime_gll;
  constexpr int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a host execution space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);
        if constexpr (store_hprime_gll) {
          if constexpr (on_device) {
            element_quadrature.hprime_gll(iz, ix) =
                quadrature.gll.hprime(iz, ix);
          } else {
            element_quadrature.hprime_gll(iz, ix) =
                quadrature.gll.h_hprime(iz, ix);
          }
        }
        if constexpr (store_weight_times_hprime_gll) {
          if constexpr (on_device) {
            element_quadrature.hprime_wgll(ix, iz) =
                quadrature.gll.hprime(iz, ix) * quadrature.gll.weights(iz);
          } else {
            element_quadrature.hprime_wgll(ix, iz) =
                quadrature.gll.h_hprime(iz, ix) * quadrature.gll.h_weights(iz);
          }
        }
      });
}

/**
 * @defgroup QuadratureDataAccess
 *
 */

/**
 * @brief Load quadrature data for a spectral element on the device
 *
 * @ingroup QuadratureDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref specfem::element::quadrature
 * @param team Team member
 * @param quadrature Quadrature data
 * @param element_quadrature Quadrature data for the element (output)
 */
template <typename MemberType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const MemberType &team,
               const specfem::compute::quadrature &quadrature,
               ViewType &element_quadrature) {

  impl_load<true>(team, quadrature, element_quadrature);
}

/**
 * @brief Load quadrature data for a spectral element on the host
 *
 * @ingroup QuadratureDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref specfem::element::quadrature
 * @param team Team member
 * @param quadrature Quadrature data
 * @param element_quadrature Quadrature data for the element (output)
 */
template <typename MemberType, typename ViewType>
void load_on_host(const MemberType &team,
                  const specfem::compute::quadrature &quadrature,
                  ViewType &element_quadrature) {
  impl_load<false>(team, quadrature, element_quadrature);
}

} // namespace compute
} // namespace specfem
