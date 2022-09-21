#ifndef QUADRATURE_H
#define QUADRATURE_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace quadrature {
/**
 * @warning GLL class is still in progress,
 * will get to it as I understand the Fortran code more
 * and what things need to be added here
 *
 */

using DeviceView1d = specfem::DeviceView1d<type_real>;
using DeviceView2d = specfem::DeviceView2d<type_real>;
using HostMirror1d = specfem::HostMirror1d<type_real>;
using HostMirror2d = specfem::HostMirror2d<type_real>;

class quadrature {
public:
  quadrature();
  quadrature(const type_real alpha, const type_real beta);
  quadrature(const type_real alpha, const type_real beta, const int N);
  void set_derivation_matrices();
  DeviceView1d get_xi() const;
  DeviceView1d get_w() const;
  DeviceView2d get_hprime() const;
  HostMirror1d get_hxi() const;
  HostMirror1d get_hw() const;
  HostMirror2d get_hhprime() const;
  int get_N() const;

private:
  type_real alpha, beta;
  int N;
  DeviceView1d xi, w;
  DeviceView2d hprime;
  HostMirror1d h_xi, h_w;
  HostMirror2d h_hprime;
  void set_allocations();
  void sync_views();
};
} // namespace quadrature

#endif
