#include "specfem/io/mesh/impl/fortran/dim3_globe/read_materials.hpp"

#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/globe_codes.hpp"
#include "specfem/medium_container.hpp"

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

  std::vector<int> region_codes(mesh.nspec), medium_codes(mesh.nspec),
      property_codes(mesh.nspec), idoubling(mesh.nspec);
  specfem::io::fortran_read_line(stream, &region_codes, &medium_codes,
                                 &property_codes, &idoubling);

  std::vector<double> rmin(mesh.nspec), rmax(mesh.nspec);
  specfem::io::fortran_read_line(stream, &rmin, &rmax);

  std::vector<bool> in_crust(mesh.nspec), in_mantle(mesh.nspec);
  specfem::io::fortran_read_line(stream, &in_crust, &in_mantle);

  material_tags tags;
  tags.medium_tags.resize(mesh.nspec);
  tags.property_tags.resize(mesh.nspec);
  auto &element_context = mesh.globe.element_context;
  element_context.resize(mesh.nspec);
  // Serial on purpose: the code translators throw on bad input, which a
  // Kokkos host parallel region cannot propagate.
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    tags.medium_tags[ispec] =
        specfem::io::mesh::impl::fortran::dim3_globe_impl::to_medium_tag(
            medium_codes[ispec]);
    tags.property_tags[ispec] =
        specfem::io::mesh::impl::fortran::dim3_globe_impl::to_property_tag(
            property_codes[ispec]);
    element_context[ispec] = {
      specfem::io::mesh::impl::fortran::dim3_globe_impl::to_region_tag(
          region_codes[ispec]),
      idoubling[ispec],
      rmin[ispec],
      rmax[ispec],
      in_crust[ispec],
      in_mantle[ispec]
    };
  }

  return tags;
}

specfem::mesh::materials<specfem::element::dimension_tag::dim3>
specfem::io::mesh::impl::fortran::dim3_globe::make_materials(
    const std::vector<specfem::element::medium_tag> &medium_tags,
    const std::vector<specfem::element::property_tag> &property_tags,
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

  // Isotropic-equivalent placeholder (lambda = mu = 1); the oracle overwrites
  // every GLL point at assembly setup.
  specfem::medium_container::material<Dimension::dim3, Medium::elastic,
                                      Property::anisotropic, Attenuation::none>
      anisotropic_elastic(1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0,
                          0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                          1.0);
  const int anisotropic_elastic_index =
      materials.add_material(anisotropic_elastic);

  std::optional<int> attenuating_elastic_index;
  if (attenuation_enabled) {
    specfem::medium_container::material<Dimension::dim3, Medium::elastic,
                                        Property::isotropic,
                                        Attenuation::constant_isotropic>
        attenuating_elastic(1.0, 1.0, 2.0, 9999.0, 9999.0, 0.0);
    attenuating_elastic_index = materials.add_material(attenuating_elastic);
  }

  for (int ispec = 0; ispec < materials.nspec; ++ispec) {
    if (property_tags[ispec] == Property::anisotropic) {
      if (medium_tags[ispec] != Medium::elastic) {
        throw std::runtime_error("Anisotropic globe elements must be elastic");
      }
      if (attenuation_enabled) {
        throw std::runtime_error("Attenuation is not implemented for 3-D "
                                 "anisotropic elastic elements");
      }
      materials.material_index_mapping[ispec] = {
        Medium::elastic, Property::anisotropic, Attenuation::none,
        anisotropic_elastic_index, ispec
      };
    } else if (property_tags[ispec] != Property::isotropic) {
      throw std::runtime_error("Unknown property tag in globe mesh database");
    } else if (medium_tags[ispec] == Medium::acoustic) {
      materials.material_index_mapping[ispec] = { Medium::acoustic,
                                                  Property::isotropic,
                                                  Attenuation::none,
                                                  acoustic_index, ispec };
    } else if (medium_tags[ispec] == Medium::elastic) {
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
