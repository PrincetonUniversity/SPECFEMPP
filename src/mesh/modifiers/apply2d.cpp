#include "jacobian/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "mesh/boundaries/acoustic_free_surface.hpp"
#include "mesh/control_nodes/control_nodes.hpp"
#include "mesh/elements/axial_elements.hpp"
#include "mesh/materials/materials.hpp"
#include "mesh/modifiers/modifiers.hpp"
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

/**
 * @brief Subdivides the boundary by adding to (index_mapping,type) the
 * sub-elements along the given edge in the given ispec_old.
 *
 * @param ispec_matspec - reference to the original ispec->material mapping to
 * reference
 * @param modifiers - reference to the modifiers object (to access subx,subz)
 * @param index_mapping - the index_mapping to append to
 * @param type - the edge type array to append to
 * @param ispec_old - ispec of the original element
 * @param edge_type - edge of the original element
 * @param ispec_old_to_new_offset - the first subelement ispec, used to compute
 * isub -> ispec_new.
 */
static void subdivide_inherit_edgecond(
    const specfem::kokkos::HostView1d<
        specfem::mesh::materials::material_specification> &ispec_matspec,
    const specfem::mesh::modifiers<specfem::dimension::type::dim2> &modifiers,
    std::vector<int> &index_mapping,
    std::vector<specfem::enums::boundaries::type> &type, const int ispec_old,
    const specfem::enums::boundaries::type edge_type,
    const int ispec_old_to_new_offset) {
  int subz, subx;
  const auto mat = ispec_matspec(ispec_old);
  std::tie(subx, subz) = modifiers.get_subdivision(mat.database_index);

  // number of subdivided elements to set
  const int nsub = (edge_type == specfem::enums::boundaries::type::TOP ||
                    edge_type == specfem::enums::boundaries::type::BOTTOM)
                       ? subx
                       : subz;
  for (int isub = 0; isub < nsub; isub++) {
    int isubz, isubx;
    switch (edge_type) {
    case specfem::enums::boundaries::type::RIGHT:
      isubz = isub;
      isubx = subx - 1;
      break;
    case specfem::enums::boundaries::type::TOP:
      isubz = subz - 1;
      isubx = isub;
      break;
    case specfem::enums::boundaries::type::LEFT:
      isubz = subz - 1;
      isubx = isub;
      break;
    case specfem::enums::boundaries::type::BOTTOM:
      isubz = subz - 1;
      isubx = isub;
      break;
    default:
      throw std::runtime_error("Edge type NONE found in specfem::mesh::mesh.");
      break;
    }
    const int ispec_new = ispec_old_to_new_offset + (subx * isubz + isubx);
    index_mapping.push_back(ispec_new);
    type.push_back(edge_type);
  }
}

static void subdivide(
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::mesh::modifiers<specfem::dimension::type::dim2> &modifiers) {
  constexpr auto DimensionType = specfem::dimension::type::dim2;
  const int nspec_old = mesh.nspec;
  const int ngnod = mesh.control_nodes.ngnod;
  const int ngnod_new = 9;
  const auto material_index_mapping_old = mesh.materials.material_index_mapping;

  /* we will expand coords, but we don't know to what size yet; use dynamic
   * resize of vector.
   */
  std::vector<std::pair<type_real, type_real> > coord_new(mesh.npgeo);
  // transfer coords to vector
  for (int i = 0; i < mesh.npgeo; i++) {
    coord_new[i] = std::make_pair(mesh.control_nodes.coord(0, i),
                                  mesh.control_nodes.coord(1, i));
  }

  // this is how we map from old ispec to new ispec
  std::vector<int> ispec_old_to_new_offsets(nspec_old);
  // const auto ispec_old_to_new = [&](const int ispec, const int isubz,
  //                                   const int isubx) {
  //   const auto mat = material_index_mapping_old(ispec);
  //   return ispec_old_to_new_offsets[ispec] +
  //          std::get<0>(modifiers.get_subdivision(mat.database_index)) * isubz
  //          + isubx;
  // };

  int nspec_new = 0; // update as we iterate
  int subz, subx;
  std::vector<int> knods_new;
  std::vector<specfem::mesh::materials::material_specification>
      material_index_mapping_new;
  for (int ispec = 0; ispec < nspec_old; ispec++) {
    const auto mat = material_index_mapping_old(ispec);
    std::tie(subx, subz) = modifiers.get_subdivision(mat.database_index);

    // form a matrix of node indices that will be used, making new ones where
    // necessary.
    const int ispec_ngnod_z = 2 * subz + 1;
    const int ispec_ngnod_x = 2 * subx + 1;
    const int total_knods = ispec_ngnod_x * ispec_ngnod_z;
    std::vector<int> ispec_knod(total_knods);
    for (int iz = 0; iz < ispec_ngnod_z; iz++) {
      for (int ix = 0; ix < ispec_ngnod_x; ix++) {
        const int ispec_inod = ispec_ngnod_x * iz + ix;
        if (iz == 0) {
          if (ix == 0) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(0, ispec);
            continue;
          } else if (ix == ispec_ngnod_x - 1) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(1, ispec);
            continue;
          } else if (ix == subx && ngnod == 9) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(4, ispec);
            continue;
          }
        } else if (iz == ispec_ngnod_z - 1) {
          if (ix == 0) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(3, ispec);
            continue;
          } else if (ix == ispec_ngnod_x - 1) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(2, ispec);
            continue;
          } else if (ix == subx && ngnod == 9) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(6, ispec);
            continue;
          }
        } else if (iz == subz && ngnod == 9) {
          if (ix == 0) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(7, ispec);
            continue;
          } else if (ix == ispec_ngnod_x - 1) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(5, ispec);
            continue;
          } else if (ix == subx && ngnod == 9) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(8, ispec);
            continue;
          }
        }
        // nod does not already exist. Make it exist
        auto shape_functions = specfem::jacobian::define_shape_functions(
            ix / ((type_real)subx) - 1, iz / ((type_real)subz) - 1, ngnod);

        double xcor = 0.0;
        double zcor = 0.0;

        for (int in = 0; in < ngnod; in++) {
          const int knod = mesh.control_nodes.knods(in, ispec);
          xcor += mesh.control_nodes.coord(0, knod) * shape_functions[in];
          zcor += mesh.control_nodes.coord(1, knod) * shape_functions[in];
        }
        ispec_knod[ispec_inod] = coord_new.size();
        coord_new.push_back(std::make_pair(xcor, zcor));
      }
    }
    ispec_old_to_new_offsets[ispec] = nspec_new;
    for (int isubz = 0; isubz < subz; isubz++) {
      for (int isubx = 0; isubx < subx; isubx++) {
        // new element (based on new ngnod=9)
        const int inod = ispec_ngnod_x * (isubz * 2) + isubx * 2;
        knods_new.push_back(ispec_knod[inod]);
        knods_new.push_back(ispec_knod[inod + 2]);
        knods_new.push_back(ispec_knod[inod + 2 + ispec_ngnod_x * 2]);
        knods_new.push_back(ispec_knod[inod + ispec_ngnod_x * 2]);
        knods_new.push_back(ispec_knod[inod + 1]);
        knods_new.push_back(ispec_knod[inod + 2 + ispec_ngnod_x]);
        knods_new.push_back(ispec_knod[inod + 1 + ispec_ngnod_x * 2]);
        knods_new.push_back(ispec_knod[inod + ispec_ngnod_x]);
        knods_new.push_back(ispec_knod[inod + 1 + ispec_ngnod_x]);
        material_index_mapping_new.push_back(mat);
        nspec_new++;
      }
    }
  }

  mesh.nspec = nspec_new;

  // skip coupled interface subdivision, since we don't need it.
  // instead force it at assembly time.
  mesh.requires_coupled_interface_recalculation = true;

  // finalize control nodes struct and materials struct
  const int npgeo_new = coord_new.size();
  mesh.npgeo = npgeo_new;
  mesh.materials.material_index_mapping = specfem::kokkos::HostView1d<
      specfem::mesh::materials::material_specification>(
      "specfem::mesh::material_index_mapping", nspec_new);
  mesh.control_nodes =
      specfem::mesh::control_nodes<specfem::dimension::type::dim2>(
          2, nspec_new, ngnod_new, coord_new.size());

  for (int igeo = 0; igeo < npgeo_new; igeo++) {
    std::tie(mesh.control_nodes.coord(0, igeo),
             mesh.control_nodes.coord(1, igeo)) = coord_new[igeo];
  }
  for (int ispec = 0; ispec < nspec_new; ispec++) {
    for (int inod = 0; inod < ngnod_new; inod++) {
      mesh.control_nodes.knods(inod, ispec) =
          knods_new[ispec * ngnod_new + inod];
    }
    mesh.materials.material_index_mapping(ispec) =
        material_index_mapping_new[ispec];
  }

  // boundaries
  { // absorbing
    std::vector<int> index_mapping;
    std::vector<specfem::enums::boundaries::type> type;
    auto &bdry = mesh.boundaries.absorbing_boundary;
    for (int ielem = 0; ielem < bdry.nelements; ielem++) {
      const int ispec = bdry.index_mapping(ielem);
      const specfem::enums::boundaries::type edge = bdry.type(ielem);
      subdivide_inherit_edgecond(material_index_mapping_old, modifiers,
                                 index_mapping, type, ispec, edge,
                                 ispec_old_to_new_offsets[ispec]);
    }
    bdry =
        specfem::mesh::absorbing_boundary<DimensionType>(index_mapping.size());
    for (int ielem = 0; ielem < bdry.nelements; ielem++) {
      bdry.index_mapping(ielem) = index_mapping[ielem];
      bdry.type(ielem) = type[ielem];
    }
  }
  { // acoustic free surface
    std::vector<int> index_mapping;
    std::vector<specfem::enums::boundaries::type> type;
    auto &bdry = mesh.boundaries.acoustic_free_surface;
    for (int ielem = 0; ielem < bdry.nelem_acoustic_surface; ielem++) {
      const int ispec = bdry.index_mapping(ielem);
      const specfem::enums::boundaries::type edge = bdry.type(ielem);
      subdivide_inherit_edgecond(material_index_mapping_old, modifiers,
                                 index_mapping, type, ispec, edge,
                                 ispec_old_to_new_offsets[ispec]);
    }
    bdry = specfem::mesh::acoustic_free_surface<DimensionType>(
        index_mapping.size());
    for (int ielem = 0; ielem < bdry.nelem_acoustic_surface; ielem++) {
      bdry.index_mapping(ielem) = index_mapping[ielem];
      bdry.type(ielem) = type[ielem];
    }
  }

  { // axial nodes
    auto &axi = mesh.axial_nodes;
    auto axiflag = axi.is_on_the_axis;
    axi = specfem::mesh::elements::axial_elements<DimensionType>(nspec_new);
    for (int ispec = 0; ispec < nspec_old; ispec++) {
      if (!axiflag(ispec)) {
        continue;
      }
      auto mat = material_index_mapping_old(ispec);
      std::tie(subx, subz) = modifiers.get_subdivision(mat.database_index);
      if (subz != 1 || subx != 1) {
        throw std::runtime_error(
            "Subdividing elements on the axis not yet supported");
      }
      axi.is_on_the_axis(ispec_old_to_new_offsets[ispec]) = true;
    }
  }

  // parameters
  mesh.parameters.ngnod = ngnod_new;
  mesh.parameters.nspec = nspec_new;
  mesh.parameters.nelemabs = mesh.boundaries.absorbing_boundary.nelements;
  mesh.parameters.nelem_acoustic_surface =
      mesh.boundaries.acoustic_free_surface.nelem_acoustic_surface;
  // mesh.parameters.nelem_acforcing = 0;
  mesh.parameters.num_fluid_solid_edges =
      mesh.coupled_interfaces.elastic_acoustic.num_interfaces;
  mesh.parameters.num_fluid_poro_edges =
      mesh.coupled_interfaces.acoustic_poroelastic.num_interfaces;
  mesh.parameters.num_solid_poro_edges =
      mesh.coupled_interfaces.elastic_poroelastic.num_interfaces;
  // mesh.parameters.nelem_on_the_axis = 0;
  // mesh.parameters.nnodes_tangential_curve = 0;

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);
}

template <>
void specfem::mesh::modifiers<specfem::dimension::type::dim2>::apply(
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) const {
  subdivide(mesh, *this);

  // forcing boundary skipped
}
