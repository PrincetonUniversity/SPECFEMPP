#include "jacobian/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

using HostView1d = specfem::kokkos::HostView1d<type_real>;
using HostView2d = specfem::kokkos::HostView2d<type_real>;

namespace {
type_real one = static_cast<type_real>(1.0);
type_real two = static_cast<type_real>(2.0);
type_real quarter = static_cast<type_real>(0.25);
type_real half = static_cast<type_real>(0.5);
} // namespace

HostView1d specfem::jacobian::define_shape_functions(const type_real xi,
                                                     const type_real gamma,
                                                     const int ngnod) {

  HostView1d h_shape2D("shape_functions::HostView::h_shape2D", ngnod);

  if (ngnod == 4) {
    //----    4-node element
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;

    //----  corner nodes
    h_shape2D(0) = quarter * sm * tm;
    h_shape2D(1) = -quarter * sp * tm;
    h_shape2D(2) = quarter * sp * tp;
    h_shape2D(3) = -quarter * sm * tp;
    type_real sumshape =
        h_shape2D(0) + h_shape2D(1) + h_shape2D(2) + h_shape2D(3);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else if (ngnod == 9) {
    //----    9-node element
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;
    const type_real s2 = xi * two;
    const type_real t2 = gamma * two;
    const type_real ss = xi * xi;
    const type_real tt = gamma * gamma;
    const type_real st = xi * gamma;

    //----  corner nodes
    h_shape2D(0) = quarter * sm * st * tm;
    h_shape2D(1) = quarter * sp * st * tm;
    h_shape2D(2) = quarter * sp * st * tp;
    h_shape2D(3) = quarter * sm * st * tp;

    //----  midside nodes
    h_shape2D(4) = half * tm * gamma * (one - ss);
    h_shape2D(5) = half * sp * xi * (one - tt);
    h_shape2D(6) = half * tp * gamma * (one - ss);
    h_shape2D(7) = half * sm * xi * (one - tt);

    //----  center node
    h_shape2D(8) = (one - ss) * (one - tt);

    type_real sumshape = h_shape2D(0) + h_shape2D(1) + h_shape2D(2) +
                         h_shape2D(3) + h_shape2D(4) + h_shape2D(5) +
                         h_shape2D(6) + h_shape2D(7) + h_shape2D(8);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }

  return h_shape2D;
}

HostView2d specfem::jacobian::define_shape_functions_derivatives(
    const type_real xi, const type_real gamma, const int ngnod) {

  HostView2d h_dershape2D("shape_functions::HostView::h_dershape2D", ndim,
                          ngnod);
  if (ngnod == 4) {
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;
    h_dershape2D(0, 0) = quarter * tm;
    h_dershape2D(0, 1) = -quarter * tm;
    h_dershape2D(0, 2) = quarter * tp;
    h_dershape2D(0, 3) = -quarter * tp;

    h_dershape2D(1, 0) = quarter * sm;
    h_dershape2D(1, 1) = -quarter * sp;
    h_dershape2D(1, 2) = quarter * sp;
    h_dershape2D(1, 3) = -quarter * sm;

    type_real dersumshape1 = h_dershape2D(0, 0) + h_dershape2D(0, 1) +
                             h_dershape2D(0, 2) + h_dershape2D(0, 3);

    type_real dersumshape2 = h_dershape2D(1, 0) + h_dershape2D(1, 1) +
                             h_dershape2D(1, 2) + h_dershape2D(1, 3);
    assert(std::abs(dersumshape1) < 1e-6);
    assert(std::abs(dersumshape2) < 1e-6);
  } else if (ngnod == 9) {
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;
    const type_real s2 = xi * two;
    const type_real t2 = gamma * two;
    const type_real ss = xi * xi;
    const type_real tt = gamma * gamma;
    const type_real st = xi * gamma;

    //----  corner nodes
    h_dershape2D(0, 0) = quarter * tm * gamma * (s2 - one);
    h_dershape2D(0, 1) = quarter * tm * gamma * (s2 + one);
    h_dershape2D(0, 2) = quarter * tp * gamma * (s2 + one);
    h_dershape2D(0, 3) = quarter * tp * gamma * (s2 - one);

    h_dershape2D(1, 0) = quarter * sm * xi * (t2 - one);
    h_dershape2D(1, 1) = quarter * sp * xi * (t2 - one);
    h_dershape2D(1, 2) = quarter * sp * xi * (t2 + one);
    h_dershape2D(1, 3) = quarter * sm * xi * (t2 + one);

    //----  midside nodes
    h_dershape2D(0, 4) = -one * st * tm;
    h_dershape2D(0, 5) = half * (one - tt) * (s2 + one);
    h_dershape2D(0, 6) = -one * st * tp;
    h_dershape2D(0, 7) = half * (one - tt) * (s2 - one);

    h_dershape2D(1, 4) = half * (one - ss) * (t2 - one);
    h_dershape2D(1, 5) = -one * st * sp;
    h_dershape2D(1, 6) = half * (one - ss) * (t2 + one);
    h_dershape2D(1, 7) = -one * st * sm;

    //----  center node
    h_dershape2D(0, 8) = -one * s2 * (one - tt);
    h_dershape2D(1, 8) = -one * t2 * (one - ss);

    type_real dersumshape1 =
        h_dershape2D(0, 0) + h_dershape2D(0, 1) + h_dershape2D(0, 2) +
        h_dershape2D(0, 3) + h_dershape2D(0, 4) + h_dershape2D(0, 5) +
        h_dershape2D(0, 6) + h_dershape2D(0, 7) + h_dershape2D(0, 8);

    type_real dersumshape2 =
        h_dershape2D(1, 0) + h_dershape2D(1, 1) + h_dershape2D(1, 2) +
        h_dershape2D(1, 3) + h_dershape2D(1, 4) + h_dershape2D(1, 5) +
        h_dershape2D(1, 6) + h_dershape2D(1, 7) + h_dershape2D(1, 8);
    assert(std::abs(dersumshape1) < 1e-6);
    assert(std::abs(dersumshape2) < 1e-6);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }

  return h_dershape2D;
}

void specfem::jacobian::define_shape_functions(HostView1d shape2D, type_real xi,
                                               const type_real gamma,
                                               const int ngnod) {

  if (ngnod == 4) {
    //----    4-node element
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;

    //----  corner nodes
    shape2D(0) = quarter * sm * tm;
    shape2D(1) = -quarter * sp * tm;
    shape2D(2) = quarter * sp * tp;
    shape2D(3) = -quarter * sm * tp;
    type_real sumshape = shape2D(0) + shape2D(1) + shape2D(2) + shape2D(3);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else if (ngnod == 9) {
    //----    9-node element
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;
    const type_real s2 = xi * two;
    const type_real t2 = gamma * two;
    const type_real ss = xi * xi;
    const type_real tt = gamma * gamma;
    const type_real st = xi * gamma;

    //----  corner nodes
    shape2D(0) = quarter * sm * st * tm;
    shape2D(1) = quarter * sp * st * tm;
    shape2D(2) = quarter * sp * st * tp;
    shape2D(3) = quarter * sm * st * tp;

    //----  midside nodes
    shape2D(4) = half * tm * gamma * (one - ss);
    shape2D(5) = half * sp * xi * (one - tt);
    shape2D(6) = half * tp * gamma * (one - ss);
    shape2D(7) = half * sm * xi * (one - tt);

    //----  center node
    shape2D(8) = (one - ss) * (one - tt);

    type_real sumshape = shape2D(0) + shape2D(1) + shape2D(2) + shape2D(3) +
                         shape2D(4) + shape2D(5) + shape2D(6) + shape2D(7) +
                         shape2D(8);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }

  return;
}

void specfem::jacobian::define_shape_functions_derivatives(
    HostView2d dershape2D, type_real xi, const type_real gamma,
    const int ngnod) {

  if (ngnod == 4) {
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;
    dershape2D(0, 0) = quarter * tm;
    dershape2D(0, 1) = -quarter * tm;
    dershape2D(0, 2) = quarter * tp;
    dershape2D(0, 3) = -quarter * tp;

    dershape2D(1, 0) = quarter * sm;
    dershape2D(1, 1) = -quarter * sp;
    dershape2D(1, 2) = quarter * sp;
    dershape2D(1, 3) = -quarter * sm;

    type_real dersumshape1 = dershape2D(0, 0) + dershape2D(0, 1) +
                             dershape2D(0, 2) + dershape2D(0, 3);

    type_real dersumshape2 = dershape2D(1, 0) + dershape2D(1, 1) +
                             dershape2D(1, 2) + dershape2D(1, 3);
    assert(std::abs(dersumshape1) < 1e-6);
    assert(std::abs(dersumshape2) < 1e-6);
  } else if (ngnod == 9) {
    const type_real sp = xi + one;
    const type_real sm = xi - one;
    const type_real tp = gamma + one;
    const type_real tm = gamma - one;
    const type_real s2 = xi * two;
    const type_real t2 = gamma * two;
    const type_real ss = xi * xi;
    const type_real tt = gamma * gamma;
    const type_real st = xi * gamma;

    //----  corner nodes
    dershape2D(0, 0) = quarter * tm * gamma * (s2 - one);
    dershape2D(0, 1) = quarter * tm * gamma * (s2 + one);
    dershape2D(0, 2) = quarter * tp * gamma * (s2 + one);
    dershape2D(0, 3) = quarter * tp * gamma * (s2 - one);

    dershape2D(1, 0) = quarter * sm * xi * (t2 - one);
    dershape2D(1, 1) = quarter * sp * xi * (t2 - one);
    dershape2D(1, 2) = quarter * sp * xi * (t2 + one);
    dershape2D(1, 3) = quarter * sm * xi * (t2 + one);

    //----  midside nodes
    dershape2D(0, 4) = -one * st * tm;
    dershape2D(0, 5) = half * (one - tt) * (s2 + one);
    dershape2D(0, 6) = -one * st * tp;
    dershape2D(0, 7) = half * (one - tt) * (s2 - one);

    dershape2D(1, 4) = half * (one - ss) * (t2 - one);
    dershape2D(1, 5) = -one * st * sp;
    dershape2D(1, 6) = half * (one - ss) * (t2 + one);
    dershape2D(1, 7) = -one * st * sm;

    //----  center node
    dershape2D(0, 8) = -one * s2 * (one - tt);
    dershape2D(1, 8) = -one * t2 * (one - ss);

    type_real dersumshape1 =
        dershape2D(0, 0) + dershape2D(0, 1) + dershape2D(0, 2) +
        dershape2D(0, 3) + dershape2D(0, 4) + dershape2D(0, 5) +
        dershape2D(0, 6) + dershape2D(0, 7) + dershape2D(0, 8);

    type_real dersumshape2 =
        dershape2D(1, 0) + dershape2D(1, 1) + dershape2D(1, 2) +
        dershape2D(1, 3) + dershape2D(1, 4) + dershape2D(1, 5) +
        dershape2D(1, 6) + dershape2D(1, 7) + dershape2D(1, 8);
    assert(std::abs(dersumshape1) < 1e-6);
    assert(std::abs(dersumshape2) < 1e-6);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }

  return;
}
