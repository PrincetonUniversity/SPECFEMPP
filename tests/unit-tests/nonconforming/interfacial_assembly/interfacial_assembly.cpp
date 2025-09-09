#include "interfacial_assembly.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem/receivers.hpp"

#include "interfacial_assembly_helper.cpp"

specfem::assembly::assembly<specfem::dimension::type::dim2>
specfem::testing::generate_interfacial_assembly(
    const interfacial_assembly_config &config) {
  constexpr specfem::dimension::type DimensionType =
      specfem::dimension::type::dim2;
  const int nspec = config.get_nspec();
  const int npgeo = config.get_npgeo();
  constexpr int ngnod = 4;
  constexpr int nproc = 1;

  specfem::mesh::mesh<specfem::dimension::type::dim2> mesh(
      npgeo, nspec, nproc, populate_controlnodes(config),
      specfem::mesh::parameters<DimensionType>(2, ngnod, nspec, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, false),
      specfem::mesh::coupled_interfaces<DimensionType>(),
      specfem::mesh::boundaries<DimensionType>(
          specfem::mesh::absorbing_boundary<DimensionType>(0),
          specfem::mesh::acoustic_free_surface<DimensionType>(0),
          specfem::mesh::forcing_boundary<DimensionType>(0)),
      specfem::mesh::tags<DimensionType>(nspec),
      specfem::mesh::elements::tangential_elements<DimensionType>(0),
      specfem::mesh::elements::axial_elements<DimensionType>(0),
      populate_materials(config));

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);
  mesh.adjacency_graph = populate_adjacency_graph(config);

  // assembly construction

  std::shared_ptr<std::vector<std::shared_ptr<
      specfem::sources::source<specfem::dimension::type::dim2> > > >
      sources = std::make_shared<std::vector<std::shared_ptr<
          specfem::sources::source<specfem::dimension::type::dim2> > > >();

  return {
    mesh,
    specfem::quadrature::quadratures(
        specfem::quadrature::gll::gll(0, 0, config.ngll)),
    *sources,
    std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::dimension::type::dim2> > >(),
    std::vector<specfem::wavefield::type>(),
    0,
    1,
    1,
    100,
    1,
    specfem::simulation::type::forward,
    false,
    nullptr
  };
}

specfem::testing::INTERFACIAL_ASSEMBLY_FIXTURE::INTERFACIAL_ASSEMBLY_FIXTURE() {
  const specfem::medium::material<specfem::element::medium_tag::elastic_psv,
                                  specfem::element::property_tag::isotropic>
      elastic_material(2500.0, 3400.0, 1963.0, 9999, 9999, 0);
  const specfem::medium::material<specfem::element::medium_tag::acoustic,
                                  specfem::element::property_tag::isotropic>
      acoustic_material(1020.0, 1500.0, 9999, 9999, 0);

  for (const auto [nelem_side1, nelem_side2, interface_len, nquad_mortar] :
       std::vector<std::tuple<int, int, type_real, int> >{
           { 10, 13, 100, 4 },
           { 10, 13, 200, 4 },
           { 10, 13, 100, 5 },
           { 15, 7, 150, 4 },
       }) {
    const specfem::testing::interfacial_assembly_config config(
        nelem_side1, 1, elastic_material, nelem_side2, 1, acoustic_material,
        interface_len, 0, false, nquad_mortar);
    this->push_back(
        std::make_pair(config, generate_interfacial_assembly(config)));
  }
}
