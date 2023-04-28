#include "jacobian/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

using HostView1d = specfem::kokkos::HostView1d<type_real>;
using HostView2d = specfem::kokkos::HostView2d<type_real>;

HostView1d specfem::jacobian::define_shape_functions(const double xi,
                                                     const double gamma,
                                                     const int ngnod) {

  HostView1d h_shape2D("shape_functions::HostView::h_shape2D", ngnod);

  if (ngnod == 4) {
    //----    4-node element
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;

    //----  corner nodes
    h_shape2D(0) = 0.25 * sm * tm;
    h_shape2D(1) = -0.25 * sp * tm;
    h_shape2D(2) = 0.25 * sp * tp;
    h_shape2D(3) = -0.25 * sm * tp;
    double sumshape = h_shape2D(0) + h_shape2D(1) + h_shape2D(2) + h_shape2D(3);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else if (ngnod == 9) {
    //----    9-node element
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;
    const double s2 = xi * 2.0;
    const double t2 = gamma * 2.0;
    const double ss = xi * xi;
    const double tt = gamma * gamma;
    const double st = xi * gamma;

    //----  corner nodes
    h_shape2D(0) = 0.25 * sm * st * tm;
    h_shape2D(1) = 0.25 * sp * st * tm;
    h_shape2D(2) = 0.25 * sp * st * tp;
    h_shape2D(3) = 0.25 * sm * st * tp;

    //----  midside nodes
    h_shape2D(4) = 0.5 * tm * gamma * (1.0 - ss);
    h_shape2D(5) = 0.5 * sp * xi * (1.0 - tt);
    h_shape2D(6) = 0.5 * tp * gamma * (1.0 - ss);
    h_shape2D(7) = 0.5 * sm * xi * (1.0 - tt);

    //----  center node
    h_shape2D(8) = (1.0 - ss) * (1.0 - tt);

    double sumshape = h_shape2D(0) + h_shape2D(1) + h_shape2D(2) +
                      h_shape2D(3) + h_shape2D(4) + h_shape2D(5) +
                      h_shape2D(6) + h_shape2D(7) + h_shape2D(8);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }

  return h_shape2D;
}

HostView2d specfem::jacobian::define_shape_functions_derivatives(
    const double xi, const double gamma, const int ngnod) {

  HostView2d h_dershape2D("shape_functions::HostView::h_dershape2D", ndim,
                          ngnod);
  if (ngnod == 4) {
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;
    h_dershape2D(0, 0) = 0.25 * tm;
    h_dershape2D(0, 1) = -0.25 * tm;
    h_dershape2D(0, 2) = 0.25 * tp;
    h_dershape2D(0, 3) = -0.25 * tp;

    h_dershape2D(1, 0) = 0.25 * sm;
    h_dershape2D(1, 1) = -0.25 * sp;
    h_dershape2D(1, 2) = 0.25 * sp;
    h_dershape2D(1, 3) = -0.25 * sm;

    double dersumshape1 = h_dershape2D(0, 0) + h_dershape2D(0, 1) +
                          h_dershape2D(0, 2) + h_dershape2D(0, 3);

    double dersumshape2 = h_dershape2D(1, 0) + h_dershape2D(1, 1) +
                          h_dershape2D(1, 2) + h_dershape2D(1, 3);
    assert(std::abs(dersumshape1) < 1e-6);
    assert(std::abs(dersumshape2) < 1e-6);
  } else if (ngnod == 9) {
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;
    const double s2 = xi * 2.0;
    const double t2 = gamma * 2.0;
    const double ss = xi * xi;
    const double tt = gamma * gamma;
    const double st = xi * gamma;

    //----  corner nodes
    h_dershape2D(0, 0) = 0.25 * tm * gamma * (s2 - 1.0);
    h_dershape2D(0, 1) = 0.25 * tm * gamma * (s2 + 1.0);
    h_dershape2D(0, 2) = 0.25 * tp * gamma * (s2 + 1.0);
    h_dershape2D(0, 3) = 0.25 * tp * gamma * (s2 - 1.0);

    h_dershape2D(1, 0) = 0.25 * sm * xi * (t2 - 1.0);
    h_dershape2D(1, 1) = 0.25 * sp * xi * (t2 - 1.0);
    h_dershape2D(1, 2) = 0.25 * sp * xi * (t2 + 1.0);
    h_dershape2D(1, 3) = 0.25 * sm * xi * (t2 + 1.0);

    //----  midside nodes
    h_dershape2D(0, 4) = -1.0 * st * tm;
    h_dershape2D(0, 5) = 0.5 * (1.0 - tt) * (s2 + 1.0);
    h_dershape2D(0, 6) = -1.0 * st * tp;
    h_dershape2D(0, 7) = 0.5 * (1.0 - tt) * (s2 - 1.0);

    h_dershape2D(1, 4) = 0.5 * (1.0 - ss) * (t2 - 1.0);
    h_dershape2D(1, 5) = -1.0 * st * sp;
    h_dershape2D(1, 6) = 0.5 * (1.0 - ss) * (t2 + 1.0);
    h_dershape2D(1, 7) = -1.0 * st * sm;

    //----  center node
    h_dershape2D(0, 8) = -1.0 * s2 * (1.0 - tt);
    h_dershape2D(1, 8) = -1.0 * t2 * (1.0 - ss);

    double dersumshape1 =
        h_dershape2D(0, 0) + h_dershape2D(0, 1) + h_dershape2D(0, 2) +
        h_dershape2D(0, 3) + h_dershape2D(0, 4) + h_dershape2D(0, 5) +
        h_dershape2D(0, 6) + h_dershape2D(0, 7) + h_dershape2D(0, 8);

    double dersumshape2 =
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

void specfem::jacobian::define_shape_functions(HostView1d shape2D, double xi,
                                               const double gamma,
                                               const int ngnod) {

  if (ngnod == 4) {
    //----    4-node element
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;

    //----  corner nodes
    shape2D(0) = 0.25 * sm * tm;
    shape2D(1) = -0.25 * sp * tm;
    shape2D(2) = 0.25 * sp * tp;
    shape2D(3) = -0.25 * sm * tp;
    double sumshape = shape2D(0) + shape2D(1) + shape2D(2) + shape2D(3);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else if (ngnod == 9) {
    //----    9-node element
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;
    const double s2 = xi * 2.0;
    const double t2 = gamma * 2.0;
    const double ss = xi * xi;
    const double tt = gamma * gamma;
    const double st = xi * gamma;

    //----  corner nodes
    shape2D(0) = 0.25 * sm * st * tm;
    shape2D(1) = 0.25 * sp * st * tm;
    shape2D(2) = 0.25 * sp * st * tp;
    shape2D(3) = 0.25 * sm * st * tp;

    //----  midside nodes
    shape2D(4) = 0.5 * tm * gamma * (1.0 - ss);
    shape2D(5) = 0.5 * sp * xi * (1.0 - tt);
    shape2D(6) = 0.5 * tp * gamma * (1.0 - ss);
    shape2D(7) = 0.5 * sm * xi * (1.0 - tt);

    //----  center node
    shape2D(8) = (1.0 - ss) * (1.0 - tt);

    double sumshape = shape2D(0) + shape2D(1) + shape2D(2) + shape2D(3) +
                      shape2D(4) + shape2D(5) + shape2D(6) + shape2D(7) +
                      shape2D(8);
    assert(std::abs(sumshape - 1) < 1e-6);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }

  return;
}

void specfem::jacobian::define_shape_functions_derivatives(
    HostView2d dershape2D, double xi, const double gamma, const int ngnod) {

  if (ngnod == 4) {
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;
    dershape2D(0, 0) = 0.25 * tm;
    dershape2D(0, 1) = -0.25 * tm;
    dershape2D(0, 2) = 0.25 * tp;
    dershape2D(0, 3) = -0.25 * tp;

    dershape2D(1, 0) = 0.25 * sm;
    dershape2D(1, 1) = -0.25 * sp;
    dershape2D(1, 2) = 0.25 * sp;
    dershape2D(1, 3) = -0.25 * sm;

    double dersumshape1 = dershape2D(0, 0) + dershape2D(0, 1) +
                          dershape2D(0, 2) + dershape2D(0, 3);

    double dersumshape2 = dershape2D(1, 0) + dershape2D(1, 1) +
                          dershape2D(1, 2) + dershape2D(1, 3);
    assert(std::abs(dersumshape1) < 1e-6);
    assert(std::abs(dersumshape2) < 1e-6);
  } else if (ngnod == 9) {
    const double sp = xi + 1.0;
    const double sm = xi - 1.0;
    const double tp = gamma + 1.0;
    const double tm = gamma - 1.0;
    const double s2 = xi * 2.0;
    const double t2 = gamma * 2.0;
    const double ss = xi * xi;
    const double tt = gamma * gamma;
    const double st = xi * gamma;

    //----  corner nodes
    dershape2D(0, 0) = 0.25 * tm * gamma * (s2 - 1.0);
    dershape2D(0, 1) = 0.25 * tm * gamma * (s2 + 1.0);
    dershape2D(0, 2) = 0.25 * tp * gamma * (s2 + 1.0);
    dershape2D(0, 3) = 0.25 * tp * gamma * (s2 - 1.0);

    dershape2D(1, 0) = 0.25 * sm * xi * (t2 - 1.0);
    dershape2D(1, 1) = 0.25 * sp * xi * (t2 - 1.0);
    dershape2D(1, 2) = 0.25 * sp * xi * (t2 + 1.0);
    dershape2D(1, 3) = 0.25 * sm * xi * (t2 + 1.0);

    //----  midside nodes
    dershape2D(0, 4) = -1.0 * st * tm;
    dershape2D(0, 5) = 0.5 * (1.0 - tt) * (s2 + 1.0);
    dershape2D(0, 6) = -1.0 * st * tp;
    dershape2D(0, 7) = 0.5 * (1.0 - tt) * (s2 - 1.0);

    dershape2D(1, 4) = 0.5 * (1.0 - ss) * (t2 - 1.0);
    dershape2D(1, 5) = -1.0 * st * sp;
    dershape2D(1, 6) = 0.5 * (1.0 - ss) * (t2 + 1.0);
    dershape2D(1, 7) = -1.0 * st * sm;

    //----  center node
    dershape2D(0, 8) = -1.0 * s2 * (1.0 - tt);
    dershape2D(1, 8) = -1.0 * t2 * (1.0 - ss);

    double dersumshape1 =
        dershape2D(0, 0) + dershape2D(0, 1) + dershape2D(0, 2) +
        dershape2D(0, 3) + dershape2D(0, 4) + dershape2D(0, 5) +
        dershape2D(0, 6) + dershape2D(0, 7) + dershape2D(0, 8);

    double dersumshape2 =
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
