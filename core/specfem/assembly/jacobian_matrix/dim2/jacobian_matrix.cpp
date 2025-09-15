#include "specfem/assembly/jacobian_matrix.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>

specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim2>::jacobian_matrix(const int nspec,
                                                     const int ngllz,
                                                     const int ngllx)
    : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
      xix("specfem::assembly::jacobian_matrix::xix", nspec, ngllz, ngllx),
      xiz("specfem::assembly::jacobian_matrix::xiz", nspec, ngllz, ngllx),
      gammax("specfem::assembly::jacobian_matrix::gammax", nspec, ngllz, ngllx),
      gammaz("specfem::assembly::jacobian_matrix::gammaz", nspec, ngllz, ngllx),
      jacobian("specfem::assembly::jacobian_matrix::jacobian", nspec, ngllz,
               ngllx),
      h_xix(specfem::kokkos::create_mirror_view(xix)),
      h_xiz(specfem::kokkos::create_mirror_view(xiz)),
      h_gammax(specfem::kokkos::create_mirror_view(gammax)),
      h_gammaz(specfem::kokkos::create_mirror_view(gammaz)),
      h_jacobian(specfem::kokkos::create_mirror_view(jacobian)) {
  return;
};

specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>::
    jacobian_matrix(
        const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh)
    : nspec(mesh.nspec), ngllz(mesh.element_grid.ngllz),
      ngllx(mesh.element_grid.ngllx),
      xix("specfem::assembly::jacobian_matrix::xix", nspec, ngllz, ngllx),
      xiz("specfem::assembly::jacobian_matrix::xiz", nspec, ngllz, ngllx),
      gammax("specfem::assembly::jacobian_matrix::gammax", nspec, ngllz, ngllx),
      gammaz("specfem::assembly::jacobian_matrix::gammaz", nspec, ngllz, ngllx),
      jacobian("specfem::assembly::jacobian_matrix::jacobian", nspec, ngllz,
               ngllx),
      h_xix(specfem::kokkos::create_mirror_view(xix)),
      h_xiz(specfem::kokkos::create_mirror_view(xiz)),
      h_gammax(specfem::kokkos::create_mirror_view(gammax)),
      h_gammaz(specfem::kokkos::create_mirror_view(gammaz)),
      h_jacobian(specfem::kokkos::create_mirror_view(jacobian)) {

  const int ngnod = mesh.ngnod;

  for (int ispec = 0; ispec < nspec; ++ispec) {

    //----- Load coorgx, coorgz in level 0 cache to be utilized later
    Kokkos::View<point::global_coordinates<specfem::dimension::type::dim2> *,
                 Kokkos::HostSpace>
        coorg("coorg", ngnod);

    // This loop is not vectorizable because access to coorg via
    // knods(ispec, in) is not vectorizable
    for (int in = 0; in < ngnod; ++in) {
      coorg(in).x = mesh.h_control_node_coord(0, ispec, in);
      coorg(in).z = mesh.h_control_node_coord(1, ispec, in);
    }

    //-----

    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ix = 0; ix < ngllx; ++ix) {

        // compute Jacobian matrix
        std::vector<std::vector<type_real> > sv_dershape2D(
            2, std::vector<type_real>(ngnod));
        for (int in = 0; in < ngnod; ++in) {
          for (int idim = 0; idim < ndim; ++idim) {
            sv_dershape2D[idim][in] = mesh.h_dshape2D(iz, ix, idim, in);
          }
        }

        auto jacobian = jacobian::compute_jacobian(coorg, ngnod, sv_dershape2D);

        this->h_xix(ispec, iz, ix) = jacobian.xix;
        this->h_gammax(ispec, iz, ix) = jacobian.gammax;
        this->h_xiz(ispec, iz, ix) = jacobian.xiz;
        this->h_gammaz(ispec, iz, ix) = jacobian.gammaz;
        this->h_jacobian(ispec, iz, ix) = jacobian.jacobian;
      }
    }
  };

  specfem::kokkos::deep_copy(xix, h_xix);
  specfem::kokkos::deep_copy(xiz, h_xiz);
  specfem::kokkos::deep_copy(gammax, h_gammax);
  specfem::kokkos::deep_copy(gammaz, h_gammaz);
  specfem::kokkos::deep_copy(jacobian, h_jacobian);

  return;
}

void specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim2>::sync_views() {
  specfem::kokkos::deep_copy(xix, h_xix);
  specfem::kokkos::deep_copy(xiz, h_xiz);
  specfem::kokkos::deep_copy(gammax, h_gammax);
  specfem::kokkos::deep_copy(gammaz, h_gammaz);
  specfem::kokkos::deep_copy(jacobian, h_jacobian);
}

std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim2>::check_small_jacobian() const {
  Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> small_jacobian(
      "specfem::assembly::jacobian_matrix::negative", nspec);

  Kokkos::deep_copy(small_jacobian, false);

  constexpr auto dimension = specfem::dimension::type::dim2;

  const type_real threshold = 1e-10;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension, true, false>;

  bool found = false;
  Kokkos::parallel_reduce(
      "specfem::assembly::jacobian_matrix::check_small_jacobian",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nspec),
      [=, *this](const int &ispec, bool &l_found) {
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int ix = 0; ix < ngllx; ++ix) {
            const specfem::point::index<dimension, false> index(ispec, iz, ix);
            const auto jacobian = [&]() {
              PointJacobianMatrixType jacobian_matrix;
              specfem::assembly::load_on_host(index, *this, jacobian_matrix);
              return jacobian_matrix.jacobian;
            }();
            if (jacobian < threshold) {
              small_jacobian(ispec) = true;
              l_found = true;
              break;
            }
          }
        }
      },
      found);
  return std::make_tuple(found, small_jacobian);
}
