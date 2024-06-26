#ifndef _COMPUTE_HPP
#define _COMPUTE_HPP

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

  specfem::compute::quadrature::GLL gll; ///< GLL object

  quadrature() = default;

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

  control_nodes() = default;
};

struct points {
  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension

  using ViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  ViewType index_mapping;                         ///< Global element
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
  int nspec;                                     ///< Number of spectral
                                                 ///< elements
  int ngllz;                                     ///< Number of quadrature
                                                 ///< points in z dimension
  int ngllx;                                     ///< Number of quadrature
                                                 ///< points in x dimension
  specfem::compute::control_nodes control_nodes; ///< Control nodes
  specfem::compute::points points;               ///< Quadrature points
  specfem::compute::quadrature quadratures;      ///< Quadrature object

  mesh() = default;

  mesh(const specfem::mesh::control_nodes &control_nodes,
       const specfem::quadrature::quadratures &quadratures);

  specfem::compute::points assemble();

  specfem::point::gcoord2 locate(const specfem::point::lcoord2 &point);

  specfem::point::lcoord2 locate(const specfem::point::gcoord2 &point);
};

template <typename MemberType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const MemberType &team,
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
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [=](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);
        if constexpr (store_hprime_gll) {
          element_quadrature.hprime_gll(iz, ix) = quadrature.gll.hprime(iz, ix);
        }
        if constexpr (store_weight_times_hprime_gll) {
          element_quadrature.hprime_wgll(ix, iz) =
              quadrature.gll.hprime(iz, ix) * quadrature.gll.weights(iz);
        }
      });
}

template <typename MemberType, typename ViewType>
void load_on_host(const MemberType &team,
                  const specfem::compute::quadrature &quadrature,
                  ViewType &element_quadrature) {

  constexpr bool store_hprime_gll = ViewType::store_hprime_gll;
  constexpr bool store_weight_times_hprime_gll =
      ViewType::store_weight_times_hprime_gll;
  constexpr int NGLL = ViewType::ngll;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [=](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);
        if constexpr (store_hprime_gll) {
          element_quadrature.hprime_gll(iz, ix) =
              quadrature.gll.h_hprime(iz, ix);
        }
        if constexpr (store_weight_times_hprime_gll) {
          element_quadrature.hprime_wgll(ix, iz) =
              quadrature.gll.h_hprime(iz, ix) * quadrature.gll.h_weights(iz);
        }
      });

  return;
}

// template <int NGLL, typename MemberType, typename MemorySpace,
//           typename MemoryTraits, bool StoreGLLQuadratureDerivatives,
//           bool WeightTimesDerivatives,
//           std::enable_if_t<std::is_same_v<typename
//           MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>,
//                            int> = 0>
// KOKKOS_FUNCTION void load_on_device(
//     const MemberType &team, const specfem::compute::quadrature &quadrature,
//     const specfem::element::quadrature<
//         NGLL, specfem::dimension::type::dim2, MemorySpace, MemoryTraits,
//         StoreGLLQuadratureDerivatives, WeightTimesDerivatives>
//         &element_quadrature) {

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, NGLL * NGLL), [=](const int &xz) {
//         int ix, iz;
//         sub2ind(xz, NGLL, iz, ix);
//         if constexpr (StoreGLLQuadratureDerivatives) {
//           element_quadrature.hprime_gll(iz, ix) = quadrature.gll.hprime(iz,
//           ix); if constexpr (WeightTimesDerivatives) {
//             element_quadrature.hprimew_gll(ix, iz) =
//                 quadrature.gll.hprime(iz, ix) * quadrature.gll.weights(iz);
//           }
//         }
//       });

//   return;
// }

// template <int NGLL, typename MemberType, typename MemorySpace,
//           typename MemoryTraits, bool StoreGLLQuadratureDerivatives,
//           bool WeightTimesDerivatives,
//           std::enable_if_t<std::is_same_v<typename
//           MemberType::execution_space::
//                                               scratch_memory_space,
//                                           MemorySpace>,
//                            int> = 0>
// void load_on_host(
//     const MemberType &team, const specfem::compute::quadrature &quadrature,
//     specfem::element::quadrature<NGLL, specfem::dimension::type::dim2,
//                                  MemorySpace, MemoryTraits,
//                                  StoreGLLQuadratureDerivatives,
//                                  WeightTimesDerivatives> &element_quadrature)
//                                  {

//   if constexpr (StoreGLLQuadratureDerivatives) {
//     Kokkos::deep_copy(element_quadrature.hprime_gll,
//     quadrature.gll.h_hprime);
//   }

//   if constexpr (WeightTimesDerivatives && StoreGLLQuadratureDerivatives) {
//     Kokkos::parallel_for(
//         Kokkos::TeamThreadRange(team, NGLL * NGLL), [=](const int &xz) {
//           int ix, iz;
//           sub2ind(xz, NGLL, iz, ix);
//           element_quadrature.hprimew_gll(iz, ix) =
//               quadrature.gll.h_hprime(iz, ix) * quadrature.gll.h_weights(iz);
//         });
//   }

//   return;
// }

} // namespace compute
} // namespace specfem

#endif
