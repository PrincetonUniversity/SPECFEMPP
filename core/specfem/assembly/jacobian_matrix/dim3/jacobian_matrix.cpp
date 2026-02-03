#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/mesh.hpp"

specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim3>::jacobian_matrix(const int nspec,
                                                     const int ngllx,
                                                     const int nglly,
                                                     const int ngllz)
    : nspec(nspec), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
      xix("specfem::assembly::jacobian_matrix::xix", nspec, ngllz, nglly,
          ngllx),
      xiy("specfem::assembly::jacobian_matrix::xiy", nspec, ngllz, nglly,
          ngllx),
      xiz("specfem::assembly::jacobian_matrix::xiz", nspec, ngllz, nglly,
          ngllx),
      etax("specfem::assembly::jacobian_matrix::etax", nspec, ngllz, nglly,
           ngllx),
      etay("specfem::assembly::jacobian_matrix::etay", nspec, ngllz, nglly,
           ngllx),
      etaz("specfem::assembly::jacobian_matrix::etaz", nspec, ngllz, nglly,
           ngllx),
      gammax("specfem::assembly::jacobian_matrix::gammax", nspec, ngllz, nglly,
             ngllx),
      gammay("specfem::assembly::jacobian_matrix::gammay", nspec, ngllz, nglly,
             ngllx),
      gammaz("specfem::assembly::jacobian_matrix::gammaz", nspec, ngllz, nglly,
             ngllx),
      jacobian("specfem::assembly::jacobian_matrix::jacobian", nspec, ngllz,
               nglly, ngllx),
      h_xix(Kokkos::create_mirror_view(xix)),
      h_xiy(Kokkos::create_mirror_view(xiy)),
      h_xiz(Kokkos::create_mirror_view(xiz)),
      h_etax(Kokkos::create_mirror_view(etax)),
      h_etay(Kokkos::create_mirror_view(etay)),
      h_etaz(Kokkos::create_mirror_view(etaz)),
      h_gammax(Kokkos::create_mirror_view(gammax)),
      h_gammay(Kokkos::create_mirror_view(gammay)),
      h_gammaz(Kokkos::create_mirror_view(gammaz)),
      h_jacobian(Kokkos::create_mirror_view(jacobian)) {}

template <typename ShapeFunctionView, typename CoordinateView>
void initialize_jacobian_matrix(
    const int nspec, const int ngllz, const int nglly, const int ngllx,
    specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>
        &jacobian_matrix,
    const ShapeFunctionView &shape_derivatives,
    const CoordinateView &coordinates) {

  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  Kokkos::parallel_for(
      "specfem::assembly::jacobian_matrix::initialize_from_mesh",
      Kokkos::MDRangePolicy<Kokkos::Rank<4> >({ 0, 0, 0, 0 },
                                              { nspec, ngllz, nglly, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int iy, const int ix) {
        // compute jacobian matrix
        const auto point_jacobian = specfem::jacobian::compute_jacobian(
            Kokkos::subview(coordinates, ispec, Kokkos::ALL(), Kokkos::ALL()),
            Kokkos::subview(shape_derivatives, iz, iy, ix, Kokkos::ALL(),
                            Kokkos::ALL()));

        specfem::assembly::store_on_device(
            specfem::point::index<dimension_tag, false>(ispec, iz, iy, ix),
            jacobian_matrix, point_jacobian);
      });

  Kokkos::fence();
}

void specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim3>::sync_views() {
  Kokkos::deep_copy(xix, h_xix);
  Kokkos::deep_copy(xiy, h_xiy);
  Kokkos::deep_copy(xiz, h_xiz);
  Kokkos::deep_copy(etax, h_etax);
  Kokkos::deep_copy(etay, h_etay);
  Kokkos::deep_copy(etaz, h_etaz);
  Kokkos::deep_copy(gammax, h_gammax);
  Kokkos::deep_copy(gammay, h_gammay);
  Kokkos::deep_copy(gammaz, h_gammaz);
  Kokkos::deep_copy(jacobian, h_jacobian);
}

std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim3>::check_small_jacobian() const {
  Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> small_jacobian(
      "specfem::assembly::jacobian_matrix::negative", nspec);

  Kokkos::deep_copy(small_jacobian, false);

  const type_real threshold = 1e-10;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension_tag, true, false>;

  bool found = false;
  Kokkos::parallel_reduce(
      "specfem::assembly::jacobian_matrix::check_small_jacobian",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nspec),
      [=, *this](const int &ispec, bool &l_found) {
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int iy = 0; iy < nglly; ++iy) {
            for (int ix = 0; ix < ngllx; ++ix) {
              // Define the local_index
              const specfem::point::index<dimension_tag, false> index(ispec, iz,
                                                                      iy, ix);

              // Get the Jacobian determinant
              const auto jacobian = [&]() {
                PointJacobianMatrixType jacobian_matrix;
                specfem::assembly::load_on_host(index, *this, jacobian_matrix);
                return jacobian_matrix.jacobian;
              }();

              // Check if below threshold
              if (jacobian < threshold) {
                small_jacobian(ispec) = true;
                l_found = true;
                break;
              }
            }
          }
        }
      },
      found);

  return std::make_tuple(found, small_jacobian);
}

specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>::
    jacobian_matrix(const specfem::assembly::mesh<dimension_tag> &assembly_mesh)
    : nspec(assembly_mesh.nspec), ngllx(assembly_mesh.element_grid.ngllx),
      nglly(assembly_mesh.element_grid.nglly),
      ngllz(assembly_mesh.element_grid.ngllz),
      xix("specfem::assembly::jacobian_matrix::xix", assembly_mesh.nspec,
          assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
          assembly_mesh.element_grid.ngllx),
      xiy("specfem::assembly::jacobian_matrix::xiy", assembly_mesh.nspec,
          assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
          assembly_mesh.element_grid.ngllx),
      xiz("specfem::assembly::jacobian_matrix::xiz", assembly_mesh.nspec,
          assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
          assembly_mesh.element_grid.ngllx),
      etax("specfem::assembly::jacobian_matrix::etax", assembly_mesh.nspec,
           assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
           assembly_mesh.element_grid.ngllx),
      etay("specfem::assembly::jacobian_matrix::etay", assembly_mesh.nspec,
           assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
           assembly_mesh.element_grid.ngllx),
      etaz("specfem::assembly::jacobian_matrix::etaz", assembly_mesh.nspec,
           assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
           assembly_mesh.element_grid.ngllx),
      gammax("specfem::assembly::jacobian_matrix::gammax", assembly_mesh.nspec,
             assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
             assembly_mesh.element_grid.ngllx),
      gammay("specfem::assembly::jacobian_matrix::gammay", assembly_mesh.nspec,
             assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
             assembly_mesh.element_grid.ngllx),
      gammaz("specfem::assembly::jacobian_matrix::gammaz", assembly_mesh.nspec,
             assembly_mesh.element_grid.ngllz, assembly_mesh.element_grid.nglly,
             assembly_mesh.element_grid.ngllx),
      jacobian("specfem::assembly::jacobian_matrix::jacobian",
               assembly_mesh.nspec, assembly_mesh.element_grid.ngllz,
               assembly_mesh.element_grid.nglly,
               assembly_mesh.element_grid.ngllx),
      h_xix(Kokkos::create_mirror_view(xix)),
      h_xiy(Kokkos::create_mirror_view(xiy)),
      h_xiz(Kokkos::create_mirror_view(xiz)),
      h_etax(Kokkos::create_mirror_view(etax)),
      h_etay(Kokkos::create_mirror_view(etay)),
      h_etaz(Kokkos::create_mirror_view(etaz)),
      h_gammax(Kokkos::create_mirror_view(gammax)),
      h_gammay(Kokkos::create_mirror_view(gammay)),
      h_gammaz(Kokkos::create_mirror_view(gammaz)),
      h_jacobian(Kokkos::create_mirror_view(jacobian)) {

  nspec = assembly_mesh.nspec;
  ngllx = assembly_mesh.element_grid.ngllx;
  nglly = assembly_mesh.element_grid.nglly;
  ngllz = assembly_mesh.element_grid.ngllz;
  const int ngnod = assembly_mesh.ngnod;

  const auto &shape_derivatives = assembly_mesh.dshape3D;
  const auto &coordinates = assembly_mesh.control_node_coordinates;

  initialize_jacobian_matrix(nspec, ngllz, nglly, ngllx, *this,
                             shape_derivatives, coordinates);

  Kokkos::deep_copy(h_xix, xix);
  Kokkos::deep_copy(h_xiy, xiy);
  Kokkos::deep_copy(h_xiz, xiz);
  Kokkos::deep_copy(h_etax, etax);
  Kokkos::deep_copy(h_etay, etay);
  Kokkos::deep_copy(h_etaz, etaz);
  Kokkos::deep_copy(h_gammax, gammax);
  Kokkos::deep_copy(h_gammay, gammay);
  Kokkos::deep_copy(h_gammaz, gammaz);
  Kokkos::deep_copy(h_jacobian, jacobian);
  return;
}
