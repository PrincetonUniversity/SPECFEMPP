#ifndef _DOMAIN_TPP
#define _DOMAIN_TPP

#include "compute/interface.hpp"
#include "domain.hpp"
#include "enumerations/interface.hpp"
#include "impl/interface.hpp"
#include "impl/receivers/interface.hpp"
#include "impl/sources/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace {
template <class medium>
void initialize_views(
    const int &nglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {

  constexpr int components = medium::components;

  Kokkos::parallel_for(
      "specfem::domain::domain::initiaze_views",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_LAMBDA(const int iglob, const int idim) {
        field(iglob, idim) = 0;
        field_dot(iglob, idim) = 0;
        field_dot_dot(iglob, idim) = 0;
        rmass_inverse(iglob, idim) = 0;
      });
}

template <class medium, class qp_type>
void initialize_rmass_inverse(
    const specfem::domain::impl::kernels::kernels<medium, qp_type> &kernels) {
  // Compute the mass matrix

  kernels.compute_mass_matrix();

  // Kokkos::fence();

  // // Invert the mass matrix
  // Kokkos::parallel_for(
  //     "specfem::domain::domain::invert_rmass_matrix",
  //     specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
  //         { 0, 0 }, { nglob, components }),
  //     KOKKOS_LAMBDA(const int iglob, const int idim) {
  //       if (rmass_inverse(iglob, idim) == 0) {
  //         rmass_inverse(iglob, idim) = 1.0;
  //       } else {
  //         rmass_inverse(iglob, idim) = 1.0 / rmass_inverse(iglob, idim);
  //       }
  //     });

  // Kokkos::fence();

  return;
}
} // namespace

template <class medium, class qp_type>
specfem::domain::domain<medium, qp_type>::domain(
    const specfem::compute::assembly &assembly,
    const qp_type &quadrature_points)
    : field(assembly.fields.forward.get_field<medium>()) {

  // this->h_field = Kokkos::create_mirror_view(this->field);
  // this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  // this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  // this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  // //----------------------------------------------------------------------------
  // // Initialize views

  // // In CUDA you can't call class lambdas inside the constructors
  // // Hence I need to use this function to initialize views
  // initialize_views<medium>(nglob, this->field, this->field_dot,
  //                          this->field_dot_dot, this->rmass_inverse);

  this->kernels = specfem::domain::impl::kernels::kernels<medium, qp_type>(
      assembly, quadrature_points);

  //----------------------------------------------------------------------------
  // Inverse of mass matrix

  initialize_rmass_inverse(this->kernels);

  return;
};

// template <class medium, class qp_type>
// void specfem::domain::domain<medium, qp_type>::sync_field(
//     specfem::sync::kind kind) {

//   if (kind == specfem::sync::DeviceToHost) {
//     Kokkos::deep_copy(h_field, field);
//   } else if (kind == specfem::sync::HostToDevice) {
//     Kokkos::deep_copy(field, h_field);
//   } else {
//     throw std::runtime_error("Could not recognize the kind argument");
//   }

//   return;
// }

// template <class medium, class qp_type>
// void specfem::domain::domain<medium, qp_type>::sync_field_dot(
//     specfem::sync::kind kind) {

//   if (kind == specfem::sync::DeviceToHost) {
//     Kokkos::deep_copy(h_field_dot, field_dot);
//   } else if (kind == specfem::sync::HostToDevice) {
//     Kokkos::deep_copy(field_dot, h_field_dot);
//   } else {
//     throw std::runtime_error("Could not recognize the kind argument");
//   }

//   return;
// }

// template <class medium, class qp_type>
// void specfem::domain::domain<medium, qp_type>::sync_field_dot_dot(
//     specfem::sync::kind kind) {

//   if (kind == specfem::sync::DeviceToHost) {
//     Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
//   } else if (kind == specfem::sync::HostToDevice) {
//     Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
//   } else {
//     throw std::runtime_error("Could not recognize the kind argument");
//   }

//   return;
// }

// template <class medium, class qp_type>
// void specfem::domain::domain<medium, qp_type>::sync_rmass_inverse(
//     specfem::sync::kind kind) {

//   if (kind == specfem::sync::DeviceToHost) {
//     Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
//   } else if (kind == specfem::sync::HostToDevice) {
//     Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
//   } else {
//     throw std::runtime_error("Could not recognize the kind argument");
//   }

//   return;
// }

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::divide_mass_matrix() {

  constexpr int components = medium::components;
  const int nglob = field.nglob;

  Kokkos::parallel_for(
      "specfem::domain::domain::divide_mass_matrix",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        field.field_dot_dot(iglob, idim) =
            field.field_dot_dot(iglob, idim) * field.mass_inverse(iglob, idim);
      });

  Kokkos::fence();

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::invert_mass_matrix() {

  constexpr int components = medium::components;
  const int nglob = field.nglob;

  Kokkos::parallel_for(
      "specfem::domain::domain::invert_mass_matrix",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        field.mass_inverse(iglob, idim) = 1.0 / field.mass_inverse(iglob, idim);
      });

  Kokkos::fence();
}

#endif /* DOMAIN_HPP_ */
