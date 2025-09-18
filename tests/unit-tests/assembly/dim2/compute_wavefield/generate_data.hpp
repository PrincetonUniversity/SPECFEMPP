// generate_data function generates wavefield data for compute_wavefield tests
// We pick a single element for different medium_tag and property_tag
// combinations and assign 1.0 to all the quadrature points in the element

#pragma once
#include "enumerations/interface.hpp"
#include "specfem/point.hpp"

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type,
          specfem::element::medium_tag medium,
          specfem::element::property_tag property>
void generate_data(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    std::vector<int> &ispecs) {

  auto field = assembly.fields.template get_simulation_field<type>();

  const int ngllx = assembly.mesh.element_grid.ngllx;
  const int ngllz = assembly.mesh.element_grid.ngllz;

  const auto elements =
      assembly.element_types.get_elements_on_host(medium, property);

  constexpr int num_components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   medium>::components;

  using PointDisplacementType =
      specfem::point::displacement<specfem::dimension::type::dim2, medium,
                                   false>;
  using PointVelocityType =
      specfem::point::velocity<specfem::dimension ::type::dim2, medium, false>;
  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2, medium,
                                   false>;

  using IndexType =
      specfem::point::index<specfem::dimension::type::dim2, false>;

  const int nelements = elements.size();

  if (nelements == 0)
    return;

  const int ispec = elements(nelements / 2);
  ispecs.push_back(ispec);

  for (int iz = 0; iz < ngllz; iz++) {
    for (int ix = 0; ix < ngllx; ix++) {
      const IndexType index(ispec, iz, ix);

      PointDisplacementType displacement(1.0);
      PointVelocityType velocity(1.0);
      PointAccelerationType acceleration(1.0);

      specfem::assembly::store_on_host(index, field, displacement, velocity,
                                       acceleration);
    }
  }

  field.copy_to_device();
}

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type>
std::vector<int> generate_data(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  std::vector<int> ispecs;

  generate_data<component, type, specfem::element::medium_tag::elastic_psv,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  generate_data<component, type, specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  return ispecs;
}
