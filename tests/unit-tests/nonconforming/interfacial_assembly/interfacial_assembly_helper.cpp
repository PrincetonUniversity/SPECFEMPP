#include "enumerations/connections.hpp"
#include "enumerations/medium.hpp"
#include "interfacial_assembly.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <type_traits>

using namespace specfem::testing;

constexpr specfem::dimension::type DimensionType =
    specfem::dimension::type::dim2;

/**
 * @brief Populates control nodes and elements for the interfacial assembly
 * configuration
 */
static specfem::mesh::control_nodes<DimensionType>
populate_controlnodes(const interfacial_assembly_config &config) {
  constexpr int ngnod = 4;
  constexpr int ndim = specfem::dimension::dimension<DimensionType>::dim;
  auto control_nodes = specfem::mesh::control_nodes<DimensionType>(
      ndim, config.get_nspec(), ngnod, config.get_npgeo());

  const int &nelem_side1 = config.nelem_side1;
  const int &nelem_side2 = config.nelem_side2;
  { // setting control node coords
    int inod = 0;

    // leftmost vertical edge
    type_real x_offset = 0;

    // side 1 cell dimensions
    type_real cell_x = config.interface_length / nelem_side1;
    type_real cell_z = cell_x / config.aspect_side1;
    for (int icol = 0; icol < nelem_side1 + 1; icol++) {
      const type_real x = x_offset + icol * cell_x;
      control_nodes.coord(0, inod) = x;
      control_nodes.coord(1, inod) = -cell_z;
      inod++;
      control_nodes.coord(0, inod) = x;
      control_nodes.coord(1, inod) = 0;
      inod++;
    }

    // side 2
    x_offset += config.interface_shift * config.interface_length;
    cell_x = config.interface_length / nelem_side2;
    cell_z = cell_x / config.aspect_side2;
    for (int icol = 0; icol < nelem_side2 + 1; icol++) {
      const type_real x = icol * cell_x;
      control_nodes.coord(0, inod) = x;
      control_nodes.coord(1, inod) = 0;
      inod++;
      control_nodes.coord(0, inod) = x;
      control_nodes.coord(1, inod) = cell_z;
      inod++;
    }
  }
  { // setting cell control nodes
    for (int ielem = 0; ielem < nelem_side1; ielem++) {
      const int ispec = ielem;
      control_nodes.knods(0, ispec) = ielem * 2;
      control_nodes.knods(1, ispec) = (ielem + 1) * 2;
      control_nodes.knods(2, ispec) = (ielem + 1) * 2 + 1;
      control_nodes.knods(3, ispec) = ielem * 2 + 1;
    }
    // index offsets for side 2
    const int ispec_off = nelem_side1;
    const int inode_off = (nelem_side1 + 1) * 2;
    for (int ielem = 0; ielem < nelem_side2; ielem++) {
      const int ispec = ielem + ispec_off;
      control_nodes.knods(0, ispec) = inode_off + ielem * 2;
      control_nodes.knods(1, ispec) = inode_off + (ielem + 1) * 2;
      control_nodes.knods(2, ispec) = inode_off + (ielem + 1) * 2 + 1;
      control_nodes.knods(3, ispec) = inode_off + ielem * 2 + 1;
    }
  }
  return control_nodes;
}

/**
 * @brief Templated helper to be delegated to by populate_materials visitor.
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
static void append_material(
    std::vector<specfem::medium::material<MediumTag, PropertyTag> >
        &material_list,
    std::vector<typename specfem::mesh::materials<
        DimensionType>::material_specification> &index_mapping,
    const specfem::medium::material<MediumTag, PropertyTag> &material) {
  const int imat = index_mapping.size();
  const int ind = material_list.size();
  material_list.push_back(material);
  index_mapping.push_back(
      specfem::mesh::materials<DimensionType>::material_specification(
          MediumTag, PropertyTag, ind, imat));
}

/**
 * @brief builds material struct for the interfacial assembly configuration
 */
static specfem::mesh::materials<DimensionType>
populate_materials(const interfacial_assembly_config &config) {
  constexpr int numat = 2;
  const int nspec = config.get_nspec();
  auto materials = specfem::mesh::materials<DimensionType>(nspec, numat);

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;
  constexpr auto elastic_psv_t = specfem::element::medium_tag::elastic_psv_t;
  constexpr auto electromagnetic_te =
      specfem::element::medium_tag::electromagnetic_te;
  constexpr auto poroelastic = specfem::element::medium_tag::poroelastic;
  constexpr auto isotropic = specfem::element::property_tag::isotropic;
  constexpr auto isotropic_cosserat =
      specfem::element::property_tag::isotropic_cosserat;
  constexpr auto anisotropic = specfem::element::property_tag::anisotropic;

  // these are the only supported materials (for now)
  std::vector<specfem::medium::material<acoustic, isotropic> >
      l_acoustic_isotropic;
  std::vector<specfem::medium::material<elastic_psv, isotropic> >
      l_elastic_psv_isotropic;
  std::vector<specfem::medium::material<elastic_sh, isotropic> >
      l_elastic_sh_isotropic;
  std::vector<specfem::mesh::materials<DimensionType>::material_specification>
      index_mapping;

  // read in each material. the materials are variants, so we need a Visitor:
  const auto add_material = [&](const auto &material) {
    using T = std::decay_t<decltype(material)>;
    if constexpr (std::is_same_v<T,
                                 decltype(l_acoustic_isotropic)::value_type>) {
      append_material(l_acoustic_isotropic, index_mapping, material);
    } else if constexpr (std::is_same_v<
                             T,
                             decltype(l_elastic_psv_isotropic)::value_type>) {
      append_material(l_elastic_psv_isotropic, index_mapping, material);
    } else if constexpr (std::is_same_v<
                             T, decltype(l_elastic_sh_isotropic)::value_type>) {
      append_material(l_elastic_sh_isotropic, index_mapping, material);
    }
  };
  std::visit(add_material, config.material_side1);
  std::visit(add_material, config.material_side2);

  // convert vector values over to materials:
  materials.get_container<acoustic, isotropic>() =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          acoustic, isotropic>(l_acoustic_isotropic.size(),
                               l_acoustic_isotropic);

  materials.get_container<elastic_psv, isotropic>() =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_psv, isotropic>(l_elastic_psv_isotropic.size(),
                                  l_elastic_psv_isotropic);

  materials.get_container<elastic_sh, isotropic>() =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_sh, isotropic>(l_elastic_sh_isotropic.size(),
                                 l_elastic_sh_isotropic);

  // set index mapping: we added side1 first, so those are indexed by zero
  for (int ispec = 0; ispec < nspec; ispec++) {
    materials.material_index_mapping(ispec) =
        index_mapping[(ispec < config.nelem_side1) ? 0 : 1];
  }

  return materials;
}

/**
 * @brief handles adjmap for the interfacial assembly configuration
 */
static specfem::mesh::adjacency_graph<DimensionType>
populate_adjacency_graph(const interfacial_assembly_config &config) {
  auto adj = specfem::mesh::adjacency_graph<DimensionType>(config.get_nspec());
  auto &graph = adj.graph();

  const int &nelem_side1 = config.nelem_side1;
  const int &nelem_side2 = config.nelem_side2;

  // loop left-right on side 1
  for (int ielem = 0; (config.make_periodic) ? (ielem < nelem_side1)
                                             : (ielem < nelem_side1 - 1);
       ielem++) {
    boost::add_edge(ielem, (ielem + 1) % nelem_side1,
                    { specfem::connections::type::strongly_conforming,
                      specfem::mesh_entity::type::right },
                    graph);
    boost::add_edge((ielem + 1) % nelem_side1, ielem,
                    { specfem::connections::type::strongly_conforming,
                      specfem::mesh_entity::type::left },
                    graph);
  }
  // loop left-right on side 2
  for (int ielem = 0; (config.make_periodic) ? (ielem < nelem_side2)
                                             : (ielem < nelem_side2 - 1);
       ielem++) {
    boost::add_edge(nelem_side1 + ielem,
                    nelem_side1 + ((ielem + 1) % nelem_side2),
                    { specfem::connections::type::strongly_conforming,
                      specfem::mesh_entity::type::right },
                    graph);
    boost::add_edge(nelem_side1 + ((ielem + 1) % nelem_side2),
                    nelem_side1 + ielem,
                    { specfem::connections::type::strongly_conforming,
                      specfem::mesh_entity::type::left },
                    graph);
  }
  // connect the two sides via a nonconforming interface
  {
    const type_real dx1 = config.interface_length / nelem_side1;
    const type_real dx2 = config.interface_length / nelem_side2;
    const type_real x2_offset =
        config.interface_shift * config.interface_length;
    const type_real eps = 1e-6 * config.interface_length;

    // iterate over side 1 and side 2, find overlaps.
    for (int ielem = 0; ielem < nelem_side1; ielem++) {
      type_real x1lo = dx1 * ielem;
      type_real x1hi = dx1 * (ielem + 1);
      for (int jelem = 0; jelem < nelem_side2; jelem++) {

        type_real x2lo = x2_offset + dx2 * jelem;
        type_real x2hi = x2_offset + dx2 * (jelem + 1);

        // try wrapping:
        if (x1lo > x2lo + config.interface_length / 2) {
          x2lo += config.interface_length;
          x2hi += config.interface_length;
        } else if (x2lo > x1lo + config.interface_length / 2) {
          x2lo -= config.interface_length;
          x2hi -= config.interface_length;
        }

        // do we have an overlap?
        if (x1lo + eps < x2hi && x2lo + eps < x1hi) {
          boost::add_edge(ielem, nelem_side1 + jelem,
                          { specfem::connections::type::nonconforming,
                            specfem::mesh_entity::type::top },
                          graph);
          boost::add_edge(nelem_side1 + jelem, ielem,
                          { specfem::connections::type::nonconforming,
                            specfem::mesh_entity::type::bottom },
                          graph);
        }
      }
    }
  }
  return adj;
}
