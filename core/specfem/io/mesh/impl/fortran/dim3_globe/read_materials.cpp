#include "specfem/io/mesh/impl/fortran/dim3_globe/read_materials.hpp"

#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/common.hpp"
#include "specfem/medium_container.hpp"

#include <Kokkos_Core.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

specfem::io::mesh::impl::fortran::dim3_globe::material_tags
specfem::io::mesh::impl::fortran::dim3_globe::read_material_tags(
    std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh) {
  specfem::io::fortran_read_line(stream, &mesh.nspec);
  if (mesh.nspec <= 0) {
    throw std::runtime_error("Globe mesh database contains no elements");
  }
  mesh.control_nodes.nspec = mesh.nspec;

  material_tags tags;
  std::vector<int> regions(mesh.nspec), idoubling(mesh.nspec);
  tags.medium_tags.resize(mesh.nspec);
  tags.property_tags.resize(mesh.nspec);
  specfem::io::fortran_read_line(stream, &regions, &tags.medium_tags,
                                 &tags.property_tags, &idoubling);

  std::vector<double> rmin(mesh.nspec), rmax(mesh.nspec);
  specfem::io::fortran_read_line(stream, &rmin, &rmax);

  std::vector<bool> in_crust(mesh.nspec), in_mantle(mesh.nspec);
  specfem::io::fortran_read_line(stream, &in_crust, &in_mantle);
  std::vector<int> in_crust_values(mesh.nspec), in_mantle_values(mesh.nspec);
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    in_crust_values[ispec] = in_crust[ispec] ? 1 : 0;
    in_mantle_values[ispec] = in_mantle[ispec] ? 1 : 0;
  }

  auto &globe = mesh.globe;
  globe.element_context.resize(mesh.nspec);
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      regions_view(regions.data(), mesh.nspec),
      idoubling_view(idoubling.data(), mesh.nspec),
      in_crust_view(in_crust_values.data(), mesh.nspec),
      in_mantle_view(in_mantle_values.data(), mesh.nspec);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      rmin_view(rmin.data(), mesh.nspec), rmax_view(rmax.data(), mesh.nspec);
  Kokkos::View<specfem::mesh::globe_element_context *, Kokkos::LayoutLeft,
               Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      element_context_view(globe.element_context.data(), mesh.nspec);
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3_globe::read_material_tags::element_context",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, mesh.nspec),
      [=](const int ispec) {
        element_context_view(
            ispec) = { regions_view(ispec),       idoubling_view(ispec),
                       rmin_view(ispec),          rmax_view(ispec),
                       in_crust_view(ispec) != 0, in_mantle_view(ispec) != 0 };
      });
  Kokkos::fence();

  return tags;
}

specfem::mesh::materials<specfem::element::dimension_tag::dim3>
specfem::io::mesh::impl::fortran::dim3_globe::make_materials(
    const std::vector<int> &medium_tags, const std::vector<int> &property_tags,
    const bool attenuation_enabled) {
  using Dimension = specfem::element::dimension_tag;
  using Medium = specfem::element::medium_tag;
  using Property = specfem::element::property_tag;
  using Attenuation = specfem::element::attenuation_tag;
  using Materials = specfem::mesh::materials<Dimension::dim3>;

  Materials materials;
  materials.nspec = static_cast<int>(medium_tags.size());
  materials.material_index_mapping.resize(materials.nspec);

  specfem::medium_container::material<Dimension::dim3, Medium::acoustic,
                                      Property::isotropic, Attenuation::none>
      acoustic(1.0, 1.0, 0.0);
  const int acoustic_index = materials.add_material(acoustic);

  specfem::medium_container::material<Dimension::dim3, Medium::elastic,
                                      Property::isotropic, Attenuation::none>
      elastic(1.0, 1.0, 2.0, 0.0);
  const int elastic_index = materials.add_material(elastic);

  std::optional<int> attenuating_elastic_index;
  if (attenuation_enabled) {
    specfem::medium_container::material<Dimension::dim3, Medium::elastic,
                                        Property::isotropic,
                                        Attenuation::constant_isotropic>
        attenuating_elastic(1.0, 1.0, 2.0, 9999.0, 9999.0, 0.0);
    attenuating_elastic_index = materials.add_material(attenuating_elastic);
  }

  for (int ispec = 0; ispec < materials.nspec; ++ispec) {
    if (property_tags[ispec] != 0) {
      throw std::runtime_error(
          "The globe database contains anisotropic/TISO elements, but "
          "SPECFEM++ has no 3-D anisotropic property container or kernel yet");
    }
    if (medium_tags[ispec] ==
        specfem::io::mesh::impl::fortran::dim3_globe_impl::medium_acoustic) {
      materials.material_index_mapping[ispec] = { Medium::acoustic,
                                                  Property::isotropic,
                                                  Attenuation::none,
                                                  acoustic_index, ispec };
    } else if (medium_tags[ispec] == specfem::io::mesh::impl::fortran::
                                         dim3_globe_impl::medium_elastic) {
      const auto attenuation = attenuation_enabled
                                   ? Attenuation::constant_isotropic
                                   : Attenuation::none;
      const int index =
          attenuation_enabled ? *attenuating_elastic_index : elastic_index;
      materials.material_index_mapping[ispec] = { Medium::elastic,
                                                  Property::isotropic,
                                                  attenuation, index, ispec };
    } else {
      throw std::runtime_error("Unknown medium tag in globe mesh database");
    }
  }
  return materials;
}
