#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mortar/fixture/mortar_fixtures.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/nonconforming_interfaces/moduled_interface_container.hpp"

namespace test_configuration::interface_containers {

using setc_type = specfem::assembly::interface::moduled_interface_container<
    specfem::dimension::type::dim2,
    specfem::assembly::interface::module::type::SINGLE_EDGE_CONTAINER>;
using detc_type = specfem::assembly::interface::moduled_interface_container<
    specfem::dimension::type::dim2,
    specfem::assembly::interface::module::type::DOUBLE_EDGE_CONTAINER>;

template <specfem::element::medium_tag MediumTag>
struct single_edge_test_container : public setc_type {
public:
  static constexpr bool is_single_edge = true;
  static constexpr specfem::element::medium_tag Medium1 = MediumTag;
  static constexpr specfem::element::medium_tag Medium2 = MediumTag;
  single_edge_test_container(
      const specfem::assembly::interface::initializer &init)
      : setc_type(init) {}
};

template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
struct double_edge_test_container : public detc_type {
public:
  static constexpr bool is_single_edge = false;
  static constexpr specfem::element::medium_tag Medium1 = MediumTag1;
  static constexpr specfem::element::medium_tag Medium2 = MediumTag2;
  double_edge_test_container(
      const specfem::assembly::interface::initializer &init)
      : detc_type(init) {}
};

using fluid_2d =
    single_edge_test_container<specfem::element::medium_tag::acoustic>;
using solid_2d =
    single_edge_test_container<specfem::element::medium_tag::elastic_psv>;

using fluid_fluid_2d =
    double_edge_test_container<specfem::element::medium_tag::acoustic,
                               specfem::element::medium_tag::acoustic>;
using fluid_solid_2d =
    double_edge_test_container<specfem::element::medium_tag::acoustic,
                               specfem::element::medium_tag::elastic_psv>;
using solid_fluid_2d =
    double_edge_test_container<specfem::element::medium_tag::elastic_psv,
                               specfem::element::medium_tag::acoustic>;
using solid_solid_2d =
    double_edge_test_container<specfem::element::medium_tag::elastic_psv,
                               specfem::element::medium_tag::elastic_psv>;

void test_on_mesh(
    const test_configuration::mesh &mesh_config,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly);

} // namespace test_configuration::interface_containers
