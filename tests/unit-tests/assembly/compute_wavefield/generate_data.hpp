// generate_data function generates wavefield data for compute_wavefield tests
// We pick a single element for different medium_tag and property_tag
// combinations and assign 1.0 to all the quadrature points in the element

#pragma once
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "point/coordinates.hpp"
#include "point/field.hpp"

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type,
          specfem::element::medium_tag medium,
          specfem::element::property_tag property>
void generate_data(specfem::compute::assembly &assembly,
                   std::vector<int> &ispecs) {

  auto field = assembly.fields.template get_simulation_field<type>();

  const int ngllx = assembly.mesh.ngllx;
  const int ngllz = assembly.mesh.ngllz;

  const auto elements =
      assembly.element_types.get_elements_on_host(medium, property);

  constexpr int num_components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   medium>::components();

  using PointFieldType =
      specfem::point::field<specfem::dimension::type::dim2, medium, true, true,
                            true, false, false>;

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

      PointFieldType point_field;

      for (int ic = 0; ic < num_components; ic++) {
        point_field.displacement(ic) = 1.0;
        point_field.velocity(ic) = 1.0;
        point_field.acceleration(ic) = 1.0;
      }

      specfem::compute::store_on_host(index, point_field, field);
    }
  }

  field.copy_to_device();
}

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type>
std::vector<int> generate_data(specfem::compute::assembly &assembly) {

  std::vector<int> ispecs;

  generate_data<component, type, specfem::element::medium_tag::elastic_sv,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  generate_data<component, type, specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  return ispecs;
}
