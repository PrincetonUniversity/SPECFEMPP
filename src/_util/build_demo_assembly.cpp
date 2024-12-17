//=====================================NOTE=====================================
// If this code is to go into the main codebase (removed from include/_util),
// make sure to clean this up. This file is a mess that should not see the
// light of day.
//===================================END NOTE===================================
#ifndef __UTIL_DEMO_ASSEMBLY_CPP_
#define __UTIL_DEMO_ASSEMBLY_CPP_

#include "_util/build_demo_assembly.hpp"

// from specfem2d.cpp
#include "compute/interface.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// end from specfem2d.cpp

#include "enumerations/simulation.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/specfem_mpi.hpp"

#include "receiver/receiver.hpp"
#include "source/source.hpp"

#include "compute/assembly/assembly.hpp"

#include "quadrature/quadrature.hpp"
#include "specfem_setup.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace _util {
namespace demo_assembly {

void construct_demo_mesh(
    specfem::mesh::mesh<DimensionType> &mesh,
    const specfem::quadrature::quadratures &quad, const int nelemx,
    const int nelemz,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals,
    int demo_construct_mode) {
  constexpr int NDIM = 2;
  constexpr type_real eps2 = 1e-12; // epsilon^2 for

  int ngllx = quad.gll.get_N();
  int ngllz = quad.gll.get_N();
  mesh.nspec = nelemx * nelemz;
  std::vector<long double> gll_xi(ngllx);
  for (int i = 0; i < ngllx; i++) {
    gll_xi[i] = quad.gll.get_hxi()(i);
  }
  // kill errors on endpoints?
  gll_xi[0] = -1;
  gll_xi[ngllx - 1] = 1;

  // generate points for mesh. Afterwards, we will build mesh fields
  std::vector<long double> offsets(nelemz);
  for (int ielemz = 0; ielemz < nelemz; ielemz++) {
    offsets[ielemz] = 0; // for now just set zero offsets

    if (demo_construct_mode = 1) { // mode == 1: half shifts every other one.
      if (ielemz % 2 == 0) {
        offsets[ielemz] = 0.5;
      }
    }
  }

  long double hz = 1.0 / nelemz;
  long double hx = 1.0 / nelemx;
  Kokkos::View<long double ****, Kokkos::HostSpace> pts(
      "temp_point_storage", mesh.nspec, ngllz, ngllx, NDIM);
  for (int ielemz = 0; ielemz < nelemz; ielemz++) {
    long double zmin = hz * ielemz;
    long double zmax = zmin + hz;

    long double z_scale = (zmax - zmin) / 2;
    long double z_center = (zmax + zmin) / 2;
    for (int ielemx = 0; ielemx < nelemx; ielemx++) {
      int ispec = ielemz * nelemx + ielemx;

      long double xmin = hx * (ielemx + offsets[ielemz]);
      long double xmax = xmin + hx;
      if (ielemx == 0)
        xmin = 0;
      if (ielemx == nelemx - 1)
        xmax = 1;
      long double x_scale = (xmax - xmin) / 2;
      long double x_center = (xmax + xmin) / 2;

      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          pts(ispec, iz, ix, 0) = x_center + x_scale * gll_xi[ix];
          pts(ispec, iz, ix, 1) = z_center + z_scale * gll_xi[iz];
        }
      }
    }
  }
  //===[ POPULATE MESH ]===

  // single material
  // 1 1 1.6e-7 2500.d0 0 0 0 9999 9999 0 0 0 0 0 0
  const type_real density = 1.0;
  const type_real cp = 1.0;
  const type_real compaction_grad = 0.0;
  const type_real Qkappa = 9999;
  const type_real Qmu = 9999;
  specfem::material::material<specfem::element::medium_tag::acoustic,
                              specfem::element::property_tag::isotropic>
      acoustic_holder(density, cp, Qkappa, Qmu, compaction_grad);

  mesh.materials = specfem::mesh::materials();
  mesh.materials.n_materials = 1;
  mesh.materials.material_index_mapping = specfem::kokkos::HostView1d<
      specfem::mesh::materials::material_specification>(
      "specfem::mesh::material_index_mapping",
      mesh.nspec); // will be populated with matspec on populate loop
  specfem::mesh::materials::material_specification matspec(
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, 0);

  std::vector<
      specfem::material::material<specfem::element::medium_tag::elastic,
                                  specfem::element::property_tag::isotropic> >
      l_elastic_isotropic(0);
  std::vector<
      specfem::material::material<specfem::element::medium_tag::acoustic,
                                  specfem::element::property_tag::isotropic> >
      l_acoustic_isotropic(1);
  l_acoustic_isotropic[0] = acoustic_holder;
  mesh.materials.elastic_isotropic = specfem::mesh::materials::material<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>(l_elastic_isotropic.size(),
                                                 l_elastic_isotropic);
  mesh.materials.acoustic_isotropic = specfem::mesh::materials::material<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>(l_acoustic_isotropic.size(),
                                                 l_acoustic_isotropic);

  // we use a trivial control node configuration: each point has a unique node
  int ngnod = 4;
  mesh.nproc = 1;
  mesh.npgeo = mesh.nspec * ngnod;

  const int num_abs_faces = 0;
  const int num_free_surf = 2 * (nelemx + nelemz);

  mesh.control_nodes = specfem::mesh::control_nodes<DimensionType>(
      NDIM, mesh.nspec, ngnod, mesh.npgeo);
  specfem::mesh::absorbing_boundary<DimensionType> bdry_abs(num_abs_faces);
  specfem::mesh::acoustic_free_surface<DimensionType> bdry_afs(num_free_surf);

  int i_free_surf = 0;

  // populate;
  for (int ielemz = 0; ielemz < nelemz; ielemz++) {
    for (int ielemx = 0; ielemx < nelemx; ielemx++) {
      int ispec = ielemz * nelemx + ielemx;
      // removals; add top and bottom
      removals.push_back(specfem::adjacency_graph::adjacency_pointer(
          ispec, specfem::enums::boundaries::type::BOTTOM));
      removals.push_back(specfem::adjacency_graph::adjacency_pointer(
          ispec, specfem::enums::boundaries::type::TOP));

      // element material
      mesh.materials.material_index_mapping(ispec) = matspec;

      // element boundaries
      if (ielemz == 0) {
        bdry_afs.index_mapping(i_free_surf) = ispec;
        bdry_afs.type(i_free_surf) = specfem::enums::boundaries::type::BOTTOM;
        i_free_surf++;
      }
      if (ielemz == nelemz - 1) {
        bdry_afs.index_mapping(i_free_surf) = ispec;
        bdry_afs.type(i_free_surf) = specfem::enums::boundaries::type::TOP;
        i_free_surf++;
      }
      if (ielemx == 0) {
        bdry_afs.index_mapping(i_free_surf) = ispec;
        bdry_afs.type(i_free_surf) = specfem::enums::boundaries::type::LEFT;
        i_free_surf++;
      }
      if (ielemx == nelemx - 1) {
        bdry_afs.index_mapping(i_free_surf) = ispec;
        bdry_afs.type(i_free_surf) = specfem::enums::boundaries::type::RIGHT;
        i_free_surf++;
      }

      // control nodes
      int off = ispec * ngnod;
      mesh.control_nodes.knods(0, ispec) = off + 0;
      for (int icomp = 0; icomp < NDIM; icomp++) {
        mesh.control_nodes.coord(icomp, off + 0) =
            (type_real)pts(ispec, 0, 0, icomp);
      }
      mesh.control_nodes.knods(1, ispec) = off + 1;
      for (int icomp = 0; icomp < NDIM; icomp++) {
        mesh.control_nodes.coord(icomp, off + 1) =
            (type_real)pts(ispec, 0, ngllx - 1, icomp);
      }
      mesh.control_nodes.knods(2, ispec) = off + 2;
      for (int icomp = 0; icomp < NDIM; icomp++) {
        mesh.control_nodes.coord(icomp, off + 2) =
            (type_real)pts(ispec, ngllz - 1, ngllx - 1, icomp);
      }
      mesh.control_nodes.knods(3, ispec) = off + 3;
      for (int icomp = 0; icomp < NDIM; icomp++) {
        mesh.control_nodes.coord(icomp, off + 3) =
            (type_real)pts(ispec, ngllz - 1, 0, icomp);
      }
    }
  }

  specfem::mesh::forcing_boundary<DimensionType> bdry_forcing(0);
  mesh.boundaries = specfem::mesh::boundaries<DimensionType>(bdry_abs, bdry_afs,
                                                             bdry_forcing);
  mesh.coupled_interfaces = specfem::mesh::coupled_interfaces<DimensionType>();

  mesh.tags =
      specfem::mesh::tags<DimensionType>(mesh.materials, mesh.boundaries);
}

void construct_demo_mesh(
    specfem::mesh::mesh<DimensionType> &mesh,
    const specfem::quadrature::quadratures &quad,
    std::vector<specfem::adjacency_graph::adjacency_pointer> &removals,
    int demo_construct_mode) {
  _util::demo_assembly::construct_demo_mesh(mesh, quad, 10, 10, removals,
                                            demo_construct_mode);
}

void construct_demo_mesh(specfem::mesh::mesh<DimensionType> &mesh,
                         const specfem::quadrature::quadratures &quad,
                         const int nelemx, const int nelemz,
                         int demo_construct_mode) {
  std::vector<specfem::adjacency_graph::adjacency_pointer> removals;
  _util::demo_assembly::construct_demo_mesh(mesh, quad, nelemx, nelemz,
                                            removals, demo_construct_mode);
}
void construct_demo_mesh(specfem::mesh::mesh<DimensionType> &mesh,
                         const specfem::quadrature::quadratures &quad,
                         int demo_construct_mode) {
  std::vector<specfem::adjacency_graph::adjacency_pointer> removals;
  _util::demo_assembly::construct_demo_mesh(mesh, quad, removals,
                                            demo_construct_mode);
}

} // namespace demo_assembly
} // namespace _util
#endif
