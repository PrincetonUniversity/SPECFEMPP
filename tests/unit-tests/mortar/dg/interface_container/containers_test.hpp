#pragma once

#include "enumerations/dimension.hpp"
#include "mortar/fixture/mortar_fixtures.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/nonconforming_interfaces/interface_modules/edge_container.hpp"

namespace test_configuration::interface_containers {

struct fluid_2d
    : public specfem::assembly::interface::module::single_edge_container<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic> {
public:
  static constexpr bool is_single_edge = true;
  fluid_2d(int num_edges)
      : specfem::assembly::interface::module::single_edge_container<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic>(num_edges) {}
};

struct solid_2d
    : public specfem::assembly::interface::module::single_edge_container<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::elastic_psv> {
public:
  static constexpr bool is_single_edge = true;
  solid_2d(int num_edges)
      : specfem::assembly::interface::module::single_edge_container<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv>(num_edges) {}
};

struct fluid_solid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::elastic_psv> {
public:
  static constexpr bool is_single_edge = false;
  fluid_solid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::elastic_psv>(num_edges1, num_edges2) {
  }
};
struct solid_fluid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::elastic_psv,
          specfem::element::medium_tag::acoustic> {
public:
  static constexpr bool is_single_edge = false;
  solid_fluid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv,
            specfem::element::medium_tag::acoustic>(num_edges1, num_edges2) {}
};

struct fluid_fluid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic> {
public:
  static constexpr bool is_single_edge = false;
  fluid_fluid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::acoustic>(num_edges1, num_edges2) {}
};
struct solid_solid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::elastic_psv,
          specfem::element::medium_tag::elastic_psv> {
public:
  static constexpr bool is_single_edge = false;
  solid_solid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv,
            specfem::element::medium_tag::elastic_psv>(num_edges1, num_edges2) {
  }
};

void test_on_mesh(
    const test_configuration::mesh &mesh_config,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly);

template <typename ContainerType>
ContainerType load_interfaces(const test_configuration::mesh &mesh_config);

} // namespace test_configuration::interface_containers
