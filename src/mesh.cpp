#include "../include/mesh.h"
#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/shape_functions.h"
#include <Kokkos_Core.hpp>
#include <algorithm>

void allocate_mesh_materials(specfem::mesh &mesh, const int nspec,
                             const int ngnod) {
  mesh.region_CPML =
      specfem::HostView1d<int>("specfem::mesh::region_CPML", nspec);
  mesh.kmato = specfem::HostView1d<int>("specfem::mesh::region_CPML", nspec);
  mesh.knods =
      specfem::HostView2d<int>("specfem::mesh::region_CPML", ngnod, nspec);

  for (int ispec = 0; ispec < nspec; ispec++) {
    mesh.kmato(ispec) = -1;
  }
  return;
}

void allocate_mesh_absorbing_boundaries(
    specfem::absorbing_boundary &abs_boundary,
    const int num_abs_boundary_faces) {
  if (num_abs_boundary_faces > 0) {
    abs_boundary.numabs = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", num_abs_boundary_faces);
    abs_boundary.abs_boundary_type = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3",
        num_abs_boundary_faces);
    abs_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4",
        num_abs_boundary_faces);
    abs_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1",
        num_abs_boundary_faces);
    abs_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2",
        num_abs_boundary_faces);
    abs_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3",
        num_abs_boundary_faces);
    abs_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4",
        num_abs_boundary_faces);
    abs_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", num_abs_boundary_faces);
    abs_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", num_abs_boundary_faces);
    abs_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", num_abs_boundary_faces);
    abs_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", num_abs_boundary_faces);
  } else {
    abs_boundary.numabs = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", 1);
    abs_boundary.abs_boundary_type = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type", 1);
    abs_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    abs_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    abs_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    abs_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    abs_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    abs_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    abs_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    abs_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    abs_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    abs_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    abs_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    abs_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
  }

  if (num_abs_boundary_faces > 0) {
    abs_boundary.codeabs =
        specfem::HostView2d<bool>("specfem::mesh::absorbing_boundary::codeabs",
                                  num_abs_boundary_faces, 4);
    abs_boundary.codeabscorner = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner",
        num_abs_boundary_faces, 4);
  } else {
    abs_boundary.codeabs = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs", 1, 1);
    abs_boundary.codeabscorner = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner", 1, 1);
  }

  if (num_abs_boundary_faces > 0) {
    for (int n = 0; n < num_abs_boundary_faces; n++) {
      abs_boundary.numabs(n) = 0;
      abs_boundary.abs_boundary_type(n) = 0;
      abs_boundary.ibegin_edge1(n) = 0;
      abs_boundary.ibegin_edge2(n) = 0;
      abs_boundary.ibegin_edge3(n) = 0;
      abs_boundary.ibegin_edge4(n) = 0;
      abs_boundary.iend_edge1(n) = 0;
      abs_boundary.iend_edge2(n) = 0;
      abs_boundary.iend_edge3(n) = 0;
      abs_boundary.iend_edge4(n) = 0;
      abs_boundary.ib_bottom(n) = 0;
      abs_boundary.ib_left(n) = 0;
      abs_boundary.ib_top(n) = 0;
      abs_boundary.ib_right(n) = 0;
      for (int i = 0; i < 4; i++) {
        abs_boundary.codeabs(n, i) = false;
        abs_boundary.codeabscorner(n, i) = false;
      }
    }
  } else {
    abs_boundary.numabs(1) = 0;
    abs_boundary.abs_boundary_type(1) = 0;
    abs_boundary.ibegin_edge1(1) = 0;
    abs_boundary.ibegin_edge2(1) = 0;
    abs_boundary.ibegin_edge3(1) = 0;
    abs_boundary.ibegin_edge4(1) = 0;
    abs_boundary.iend_edge1(1) = 0;
    abs_boundary.iend_edge2(1) = 0;
    abs_boundary.iend_edge3(1) = 0;
    abs_boundary.iend_edge4(1) = 0;
    abs_boundary.ib_bottom(1) = 0;
    abs_boundary.ib_left(1) = 0;
    abs_boundary.ib_top(1) = 0;
    abs_boundary.ib_right(1) = 0;
    abs_boundary.codeabs(1, 1) = false;
    abs_boundary.codeabscorner(1, 1) = false;
  }
  return;
}

void allocate_mesh_acforcing_boundaries(
    specfem::forcing_boundary &acforcing_boundary,
    const int nelement_acforcing) {
  if (nelement_acforcing > 0) {
    acforcing_boundary.numacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    acforcing_boundary.codeacforcing = specfem::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing, 4);
    acforcing_boundary.typeacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    acforcing_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", nelement_acforcing);
    acforcing_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", nelement_acforcing);
    acforcing_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", nelement_acforcing);
    acforcing_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", nelement_acforcing);
    acforcing_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", nelement_acforcing);
    acforcing_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", nelement_acforcing);
    acforcing_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", nelement_acforcing);
    acforcing_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", nelement_acforcing);
    acforcing_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", nelement_acforcing);
    acforcing_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", nelement_acforcing);
    acforcing_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", nelement_acforcing);
    acforcing_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", nelement_acforcing);
  } else {
    acforcing_boundary.numacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    acforcing_boundary.codeacforcing = specfem::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", 1, 1);
    acforcing_boundary.typeacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    acforcing_boundary.ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    acforcing_boundary.ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    acforcing_boundary.ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    acforcing_boundary.ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    acforcing_boundary.iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    acforcing_boundary.iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    acforcing_boundary.iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    acforcing_boundary.iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    acforcing_boundary.ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    acforcing_boundary.ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    acforcing_boundary.ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    acforcing_boundary.ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
  }

  if (nelement_acforcing > 0) {
    for (int n = 0; n < nelement_acforcing; n++) {
      acforcing_boundary.numacforcing(n) = 0;
      acforcing_boundary.typeacforcing(n) = 0;
      acforcing_boundary.ibegin_edge1(n) = 0;
      acforcing_boundary.ibegin_edge2(n) = 0;
      acforcing_boundary.ibegin_edge3(n) = 0;
      acforcing_boundary.ibegin_edge4(n) = 0;
      acforcing_boundary.iend_edge1(n) = 0;
      acforcing_boundary.iend_edge2(n) = 0;
      acforcing_boundary.iend_edge3(n) = 0;
      acforcing_boundary.iend_edge4(n) = 0;
      acforcing_boundary.ib_bottom(n) = 0;
      acforcing_boundary.ib_left(n) = 0;
      acforcing_boundary.ib_top(n) = 0;
      acforcing_boundary.ib_right(n) = 0;
      for (int i = 0; i < 4; i++) {
        acforcing_boundary.codeacforcing(n, i) = false;
      }
    }
  } else {
    acforcing_boundary.numacforcing(1) = 0;
    acforcing_boundary.typeacforcing(1) = 0;
    acforcing_boundary.ibegin_edge1(1) = 0;
    acforcing_boundary.ibegin_edge2(1) = 0;
    acforcing_boundary.ibegin_edge3(1) = 0;
    acforcing_boundary.ibegin_edge4(1) = 0;
    acforcing_boundary.iend_edge1(1) = 0;
    acforcing_boundary.iend_edge2(1) = 0;
    acforcing_boundary.iend_edge3(1) = 0;
    acforcing_boundary.iend_edge4(1) = 0;
    acforcing_boundary.ib_bottom(1) = 0;
    acforcing_boundary.ib_left(1) = 0;
    acforcing_boundary.ib_top(1) = 0;
    acforcing_boundary.ib_right(1) = 0;
    acforcing_boundary.codeacforcing(1, 1) = false;
  }
  return;
}

void allocate_acfree_surface(specfem::acoustic_free_surface &acfree_surface,
                             const int nelem_acoustic_surface) {
  if (nelem_acoustic_surface > 0) {
    acfree_surface.numacfree_surface = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::numacfree_surface",
        nelem_acoustic_surface);
    acfree_surface.typeacfree_surface = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::typeacfree_surface",
        nelem_acoustic_surface);
    acfree_surface.e1 = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::e1", nelem_acoustic_surface);
    acfree_surface.e2 = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::e2", nelem_acoustic_surface);
    acfree_surface.ixmin = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ixmin", nelem_acoustic_surface);
    acfree_surface.ixmax = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ixmax", nelem_acoustic_surface);
    acfree_surface.izmin = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::izmin", nelem_acoustic_surface);
    acfree_surface.izmax = specfem::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::izmax", nelem_acoustic_surface);
  }
  return;
}

void allocate_tangential_elements(
    specfem::tangential_elements &tangential_nodes,
    const int nnodes_tangential_curve) {
  if (nnodes_tangential_curve > 0) {
    tangential_nodes.x = specfem::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::x", nnodes_tangential_curve);
    tangential_nodes.y = specfem::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::y", nnodes_tangential_curve);
  } else {
    tangential_nodes.x =
        specfem::HostView1d<type_real>("specfem::mesh::tangential_nodes::x", 1);
    tangential_nodes.y =
        specfem::HostView1d<type_real>("specfem::mesh::tangential_nodes::y", 1);
  }

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      tangential_nodes.x(inum) = 0.0;
      tangential_nodes.y(inum) = 0.0;
    }
  } else {
    tangential_nodes.x(1) = 0.0;
    tangential_nodes.y(1) = 0.0;
  }
  return;
}

void allocate_axial_elements(specfem::axial_elements &axial_nodes,
                             const int nspec) {
  axial_nodes.is_on_the_axis = specfem::HostView1d<bool>(
      "specfem::mesh::axial_element::is_on_the_axis", nspec);

  for (int inum = 0; inum < nspec; inum++) {
    axial_nodes.is_on_the_axis(nspec) = false;
  }

  return;
}

void specfem::mesh::allocate() {

  allocate_mesh_materials(*this, this->properties.nspec,
                          this->properties.ngnod);
  allocate_mesh_absorbing_boundaries(this->abs_boundary,
                                     this->properties.nelemabs);
  allocate_mesh_acforcing_boundaries(this->acforcing_boundary,
                                     this->properties.nelem_acforcing);
  allocate_acfree_surface(this->acfree_surface,
                          this->properties.nelem_acoustic_surface);
  allocate_tangential_elements(this->tangential_nodes,
                               this->properties.nnodes_tangential_curve);
  allocate_axial_elements(this->axial_nodes, this->properties.nspec);
  return;
}

struct qp {
  type_real x = 0, y = 0;
  int iloc = 0, iglob = 0;
};

specfem::HostView3d<int>
assign_numbering(const specfem::HostView2d<type_real> coorg,
                 const specfem::HostView2d<int> knods,
                 const quadrature::quadrature &quadx,
                 const quadrature::quadrature &quadz) {

  int ngnod = knods.extent(0);
  int nspec = knods.extent(1);

  int ngllx = quadx.get_N();
  int ngllz = quadz.get_N();
  int ngllxz = ngllx * ngllz;

  specfem::HostMirror1d<type_real> xi = quadx.get_hxi();
  specfem::HostMirror1d<type_real> gamma = quadz.get_hxi();

  specfem::HostView3d<type_real> shape2D("specfem::mesh::assign_numbering",
                                         ngllz, ngllx, ngnod);

  std::vector<qp> cart_cord(nspec * ngllxz);
  std::vector<qp> *pcart_cord = &cart_cord;
  int scratch_size =
      2 * specfem::HostScratchView1d<type_real>::shmem_size(ngnod);

  // Allocate shape functions
  Kokkos::parallel_for(
      "shape_functions", specfem::HostMDrange<2>({ 0, 0 }, { ngllz, ngllx }),
      KOKKOS_LAMBDA(const int iz, const int ix) {
        type_real ixxi = xi(ix);
        type_real izgamma = gamma(iz);

        specfem::HostView1d<type_real> shape2D_tmp =
            shape_functions::define_shape_functions(ixxi, izgamma, ngnod);
        for (int in = 0; in < ngnod; in++)
          shape2D(iz, ix, in) = shape2D_tmp(in);
      });

  // Calculate the x and y coordinates for every GLL point

  Kokkos::parallel_for(
      specfem::HostTeam(nspec, Kokkos::AUTO, ngnod)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const specfem::HostTeam::member_type &teamMember) {
        const int ispec = teamMember.league_rank();

        //----- Load coorgx, coorgz in level 0 cache to be utilized later
        specfem::HostScratchView1d<type_real> s_coorgx(
            teamMember.team_scratch(0), ngnod);
        specfem::HostScratchView1d<type_real> s_coorgz(
            teamMember.team_scratch(0), ngnod);

        // This loop is not vectorizable because access to coorg via
        // knods(ispec, in) is not vectorizable
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, ngnod),
                             [&](const int in) {
                               s_coorgx(in) = coorg(0, knods(in, ispec));
                               s_coorgz(in) = coorg(1, knods(in, ispec));
                             });

        teamMember.team_barrier();
        //-----

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              const int iloc = ispec * (ngllxz) + xz;

              type_real xcor = 0.0;
              type_real ycor = 0.0;

              // FIXME:: Multi reduction is not yet implemented in kokkos
              // This is hacky way of doing this using double vector loops
              // Use multiple reducers once kokkos enables the feature
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(teamMember, ngnod),
                  [&](const int &in, type_real &update_xcor) {
                    update_xcor += shape2D(iz, ix, in) * s_coorgx(in);
                  },
                  xcor);
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(teamMember, ngnod),
                  [&](const int &in, type_real &update_ycor) {
                    update_ycor += shape2D(iz, ix, in) * s_coorgz(in);
                  },
                  ycor);
              // ------------
              // Hacky way of doing this (but nacessary), because KOKKOS_LAMBDA
              // is a const operation so I cannot update cart_cord inside of a
              // lambda directly Since iloc is different within every thread I
              // ensure that I don't have a race condition here.
              (*pcart_cord)[iloc].x = xcor;
              (*pcart_cord)[iloc].y = ycor;
              (*pcart_cord)[iloc].iloc = iloc;
            });
      });

  // Sort cartesian coordinates in ascending order i.e.
  // cart_cord = [{0,0}, {0, 25}, {0, 50}, ..., {50, 0}, {50, 25}, {50, 50}]
  std::sort(cart_cord.begin(), cart_cord.end(),
            [&](const qp qp1, const qp qp2) {
              if (qp1.x != qp2.x) {
                return qp1.x < qp2.x;
              }

              return qp1.y < qp2.y;
            });

  // Setup numbering
  int ig = 0;
  cart_cord[0].iglob = ig;

  for (int iloc = 1; iloc < cart_cord.size(); iloc++) {
    // check if the previous point is same as current
    if (((cart_cord[iloc].x * cart_cord[iloc].x -
          cart_cord[iloc - 1].x * cart_cord[iloc - 1].x) +
         (cart_cord[iloc].y * cart_cord[iloc].y -
          cart_cord[iloc - 1].y * cart_cord[iloc - 1].y)) > 1e-6) {
      ig++;
    }
    cart_cord[iloc].iglob = ig;
  }

  std::vector<qp> copy_cart_cord(nspec * ngllxz);

  // reorder cart cord in original format
  for (int i = 0; i < cart_cord.size(); i++) {
    int iloc = cart_cord[i].iloc;
    copy_cart_cord[iloc] = cart_cord[i];
  }

  int nglob = ig;

  // Assign numbering to corresponding ispec, iz, ix
  specfem::HostView3d<int> ibool("specfem::mesh::ibool", nspec, ngllz, ngllx);
  std::vector<int> iglob_counted(nglob, -1);
  int iloc = 0;
  int inum = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        if (iglob_counted[copy_cart_cord[iloc].iglob] == -1) {
          ibool(ispec, iz, ix) = inum;
          iglob_counted[copy_cart_cord[iloc].iglob] = inum;
          inum++;
        } else {
          ibool(ispec, iz, ix) = iglob_counted[copy_cart_cord[iloc].iglob];
        }
        iloc++;
      }
    }
  }

  assert(inum == nglob);

  return ibool;
}

void specfem::mesh::setup(const quadrature::quadrature &quadx,
                          const quadrature::quadrature &quadz) {

  this->ibool = assign_numbering(this->coorg, this->knods, quadx, quadz);
}
