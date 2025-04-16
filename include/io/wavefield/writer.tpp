#pragma once

#include "io/wavefield/writer.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "utilities/strings.hpp"

template <typename OutputLibrary>
specfem::io::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::io::wavefield_writer<OutputLibrary>::write(
    specfem::compute::assembly &assembly) {
  auto &forward = assembly.fields.forward;
  auto &boundary_values = assembly.boundary_values;
  auto &element_types = assembly.element_types;
  auto &mesh = assembly.mesh;

  using DomainView =
      Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  forward.copy_to_host();
  boundary_values.copy_to_host();

  std::string dst = this->output_folder + "/ForwardWavefield";

  if (this->istep != -1) {
    dst += specfem::utilities::to_zero_lead(istep, 6);
  }

  typename OutputLibrary::File file(dst);

  typename OutputLibrary::Group elastic_psv = file.createGroup("/ElasticSV");
  const auto &elastic_psv_field =
      forward.get_field<specfem::element::medium_tag::elastic_psv>();
  typename OutputLibrary::Group elastic_sh = file.createGroup("/ElasticSH");
  const auto &elastic_sh_field =
      forward.get_field<specfem::element::medium_tag::elastic_sh>();
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");
  const auto &acoustic_field =
      forward.get_field<specfem::element::medium_tag::acoustic>();
  typename OutputLibrary::Group poroelastic = file.createGroup("/Poroelastic");
  const auto &poroelastic_field =
      forward.get_field<specfem::element::medium_tag::poroelastic>();
  typename OutputLibrary::Group boundary = file.createGroup("/Boundary");
  typename OutputLibrary::Group stacey = boundary.createGroup("/Stacey");

  elastic_psv.createDataset("Displacement", elastic_psv_field.h_field).write();
  elastic_psv.createDataset("Velocity", elastic_psv_field.h_field_dot).write();
  elastic_psv.createDataset("Acceleration", elastic_psv_field.h_field_dot_dot)
      .write();

  {
    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic_psv);
    const int n_elements = element_indices.size();
    DomainView x("xcoordinates", elastic_psv_field.h_field.extent(0));
    DomainView z("zcoordinates", elastic_psv_field.h_field.extent(0));
    for (int i = 0; i < n_elements; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          const int iglob = forward.template get_iglob<false>(i, iz, ix, specfem::element::medium_tag::elastic_psv);
          x(iglob) = mesh.points.h_coord(0, ispec, iz, ix);
          z(iglob) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }
    elastic_psv.createDataset("X", x).write();
    elastic_psv.createDataset("Z", z).write();
  }

  elastic_sh.createDataset("Displacement", elastic_sh_field.h_field).write();
  elastic_sh.createDataset("Velocity", elastic_sh_field.h_field_dot).write();
  elastic_sh.createDataset("Acceleration", elastic_sh_field.h_field_dot_dot)
      .write();

  {
    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic_sh);
    const int n_elements = element_indices.size();
    DomainView x("xcoordinates", elastic_sh_field.h_field.extent(0));
    DomainView z("zcoordinates", elastic_sh_field.h_field.extent(0));
    for (int i = 0; i < n_elements; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          const int iglob = forward.template get_iglob<false>(i, iz, ix, specfem::element::medium_tag::elastic_sh);
          x(iglob) = mesh.points.h_coord(0, ispec, iz, ix);
          z(iglob) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }
    elastic_sh.createDataset("X", x).write();
    elastic_sh.createDataset("Z", z).write();
  }

  acoustic.createDataset("Potential", acoustic_field.h_field).write();
  acoustic.createDataset("PotentialDot", acoustic_field.h_field_dot).write();
  acoustic.createDataset("PotentialDotDot", acoustic_field.h_field_dot_dot).write();

  {
    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::acoustic);
    const int n_elements = element_indices.size();
    DomainView x("xcoordinates", acoustic_field.h_field.extent(0));
    DomainView z("zcoordinates", acoustic_field.h_field.extent(0));
    for (int i = 0; i < n_elements; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          const int iglob = forward.template get_iglob<false>(i, iz, ix, specfem::element::medium_tag::acoustic);
          x(iglob) = mesh.points.h_coord(0, ispec, iz, ix);
          z(iglob) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }
    acoustic.createDataset("X", x).write();
    acoustic.createDataset("Z", z).write();
  }

  poroelastic.createDataset("Displacement", poroelastic_field.h_field).write();
  poroelastic.createDataset("Velocity", poroelastic_field.h_field_dot).write();
  poroelastic.createDataset("Acceleration", poroelastic_field.h_field_dot_dot)
      .write();

  {
    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::poroelastic);
    const int n_elements = element_indices.size();
    DomainView x("xcoordinates", poroelastic_field.h_field.extent(0));
    DomainView z("zcoordinates", poroelastic_field.h_field.extent(0));
    for (int i = 0; i < n_elements; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          const int iglob = forward.template get_iglob<false>(i, iz, ix, specfem::element::medium_tag::poroelastic);
          x(iglob) = mesh.points.h_coord(0, ispec, iz, ix);
          z(iglob) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }
    poroelastic.createDataset("X", x).write();
    poroelastic.createDataset("Z", z).write();
  }

  stacey
      .createDataset("IndexMapping",
                     boundary_values.stacey.h_property_index_mapping)
      .write();
  stacey
      .createDataset("ElasticAcceleration",
                     boundary_values.stacey.elastic.h_values)
      .write();
  stacey
      .createDataset("AcousticAcceleration",
                     boundary_values.stacey.acoustic.h_values)
      .write();
  stacey
      .createDataset("PoroelasticAcceleration",
                     boundary_values.stacey.poroelastic.h_values)
      .write();

  std::cout << "Wavefield written to " << dst
            << std::endl;
}
