// generate_data function generates wavefield data for compute_wavefield tests
// We pick a single element for different medium_tag and property_tag
// combinations and assign 1.0 to all the quadrature points in the element

#pragma once
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "point/coordinates.hpp"
#include "point/field.hpp"

namespace impl {
template <specfem::element::medium_tag MediumTag,
          specfem::wavefield::component Component>
class field_type_parameters;

template <specfem::element::medium_tag MediumTag>
class field_type_parameters<MediumTag,
                            specfem::wavefield::component::displacement> {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto store_displacement = true;
  constexpr static auto store_velocity = false;
  constexpr static auto store_acceleration = false;
  constexpr static auto store_mass_matrix = false;
  constexpr static auto num_components = 2;
};

template <specfem::element::medium_tag MediumTag>
class field_type_parameters<MediumTag,
                            specfem::wavefield::component::velocity> {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto store_displacement = false;
  constexpr static auto store_velocity = true;
  constexpr static auto store_acceleration = false;
  constexpr static auto store_mass_matrix = false;
  constexpr static auto num_components = 2;
};

template <specfem::element::medium_tag MediumTag>
class field_type_parameters<MediumTag,
                            specfem::wavefield::component::acceleration> {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto store_displacement = false;
  constexpr static auto store_velocity = false;
  constexpr static auto store_acceleration = true;
  constexpr static auto store_mass_matrix = false;
  constexpr static auto num_components = 2;
};
} // namespace impl

template <specfem::wavefield::component component,
          specfem::wavefield::type type, specfem::element::medium_tag medium,
          specfem::element::property_tag property>
void generate_data(specfem::compute::assembly &assembly,
                   std::vector<int> &ispecs) {

  auto field = assembly.fields.template get_simulation_field<type>();

  const int ngllx = assembly.mesh.ngllx;
  const int ngllz = assembly.mesh.ngllz;

  const auto elements =
      assembly.properties.get_elements_on_host(medium, property);
  using field_parameters = impl::field_type_parameters<medium, component>;

  constexpr int num_components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   medium>::components();

  using PointFieldType =
      specfem::point::field<specfem::dimension::type::dim2, medium,
                            field_parameters::store_displacement,
                            field_parameters::store_velocity,
                            field_parameters::store_acceleration,
                            field_parameters::store_mass_matrix, false>;

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
        point_field(ic) = 1.0;
      }

      specfem::compute::store_on_host(index, point_field, field);
    }
  }

  field.copy_to_device();
}

template <specfem::wavefield::component component,
          specfem::wavefield::type type>
std::vector<int> generate_data(specfem::compute::assembly &assembly) {

  std::vector<int> ispecs;

  generate_data<component, type, specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  generate_data<component, type, specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  return ispecs;
}
