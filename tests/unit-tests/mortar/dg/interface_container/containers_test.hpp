#pragma once

#include "enumerations/dimension.hpp"
#include "mortar/fixture/mortar_fixtures.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/nonconforming_interfaces/interface_modules/edge_container.hpp"

namespace test_configuration::interface_containers {

struct fluid_2d
    : public specfem::assembly::interface::module::single_edge_container<
          specfem::dimension::type::dim2> {
public:
  static constexpr bool is_single_edge = true;
  static constexpr specfem::element::medium_tag Medium1 =
      specfem::element::medium_tag::acoustic;
  static constexpr specfem::element::medium_tag Medium2 =
      specfem::element::medium_tag::acoustic;
  fluid_2d(int num_edges)
      : specfem::assembly::interface::module::single_edge_container<
            specfem::dimension::type::dim2>(num_edges) {}
};

struct solid_2d
    : public specfem::assembly::interface::module::single_edge_container<
          specfem::dimension::type::dim2> {
public:
  static constexpr bool is_single_edge = true;
  static constexpr specfem::element::medium_tag Medium1 =
      specfem::element::medium_tag::elastic_psv;
  static constexpr specfem::element::medium_tag Medium2 =
      specfem::element::medium_tag::elastic_psv;
  solid_2d(int num_edges)
      : specfem::assembly::interface::module::single_edge_container<
            specfem::dimension::type::dim2>(num_edges) {}
};

struct fluid_solid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2> {
public:
  static constexpr bool is_single_edge = false;
  static constexpr specfem::element::medium_tag Medium1 =
      specfem::element::medium_tag::acoustic;
  static constexpr specfem::element::medium_tag Medium2 =
      specfem::element::medium_tag::elastic_psv;
  fluid_solid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2>(num_edges1, num_edges2) {}
};
struct solid_fluid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2> {
public:
  static constexpr bool is_single_edge = false;
  static constexpr specfem::element::medium_tag Medium1 =
      specfem::element::medium_tag::elastic_psv;
  static constexpr specfem::element::medium_tag Medium2 =
      specfem::element::medium_tag::acoustic;
  solid_fluid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2>(num_edges1, num_edges2) {}
};

struct fluid_fluid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2> {
public:
  static constexpr bool is_single_edge = false;
  static constexpr specfem::element::medium_tag Medium1 =
      specfem::element::medium_tag::acoustic;
  static constexpr specfem::element::medium_tag Medium2 =
      specfem::element::medium_tag::acoustic;
  fluid_fluid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2>(num_edges1, num_edges2) {}
};
struct solid_solid_2d
    : public specfem::assembly::interface::module::double_edge_container<
          specfem::dimension::type::dim2> {
public:
  static constexpr bool is_single_edge = false;
  static constexpr specfem::element::medium_tag Medium1 =
      specfem::element::medium_tag::elastic_psv;
  static constexpr specfem::element::medium_tag Medium2 =
      specfem::element::medium_tag::elastic_psv;
  solid_solid_2d(int num_edges1, int num_edges2)
      : specfem::assembly::interface::module::double_edge_container<
            specfem::dimension::type::dim2>(num_edges1, num_edges2) {}
};

void test_on_mesh(
    const test_configuration::mesh &mesh_config,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly);

template <typename ContainerType>
ContainerType load_interfaces(const test_configuration::mesh &mesh_config);

} // namespace test_configuration::interface_containers
