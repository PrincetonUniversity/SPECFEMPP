#pragma once

#include "specfem/assembly.hpp"
#include <gtest/gtest.h>

#include <variant>

namespace specfem::testing {
struct interfacial_assembly_config;

/**
 * @brief Builds an assembly modeling a (optionally) periodic nonconforming
 * interface. The interface is flat and extends only one cell from the
 * interface.
 *
 * `nelem_side1`  ...                     `nelem_side1` + `nelem_side2` - 1
 * -----------------------------------------
 * /    |    |    |    |    |    |    |    /
 * -----------------------------------------
 * //       |       |       |       |     //
 * -----------------------------------------
 *    0        1       ...      `nelem_side1` - 1
 *
 * The domain is set with a length of `interface_length`. Each cell has an
 * aspect ratio (width / height) set by `aspect_side<1/2>` respectively.
 * The left and right sides have "conforming" corner nodes, unless a nonzero
 * `interface_shift` value is provided, where the cells on the top are shifted
 * by a distance `interface_shift` * `interface_length`. (not yet supported)
 *
 * This assembly is meant to test boundary integral computations.
 */
specfem::assembly::assembly<specfem::dimension::type::dim2>
generate_interfacial_assembly(const interfacial_assembly_config &config);

/**
 * @brief generate_interfacial_assembly overloads by passing in the config
 * constructor arguments.
 */
template <typename... Args>
specfem::assembly::assembly<specfem::dimension::type::dim2>
generate_interfacial_assembly(Args... args) {
  return generate_interfacial_assembly(interfacial_assembly_config(args...));
}

struct interfacial_assembly_config {
  int nelem_side1;
  type_real aspect_side1;
  int nelem_side2;
  type_real aspect_side2;
  type_real interface_length;
  type_real interface_shift;
  bool make_periodic;
  int ngll;
  using MaterialType = std::variant<
      specfem::medium::material<specfem::element::medium_tag::acoustic,
                                specfem::element::property_tag::isotropic>,
      specfem::medium::material<specfem::element::medium_tag::elastic_psv,
                                specfem::element::property_tag::isotropic>,
      specfem::medium::material<specfem::element::medium_tag::elastic_sh,
                                specfem::element::property_tag::isotropic> >;
  MaterialType material_side1;
  MaterialType material_side2;

  interfacial_assembly_config(
      const int nelem_side1, const type_real aspect_side1,
      const MaterialType material_side1, const int nelem_side2,
      const type_real aspect_side2, const MaterialType material_side2,
      const type_real interface_length, const type_real interface_shift,
      const bool make_periodic, const int ngll)
      : nelem_side1(nelem_side1), aspect_side1(aspect_side1),
        material_side1(material_side1), nelem_side2(nelem_side2),
        aspect_side2(aspect_side2), material_side2(material_side2),
        interface_length(interface_length), interface_shift(interface_shift),
        make_periodic(make_periodic), ngll(ngll) {
    this->interface_shift = 0; // not yet supported
  }

  int get_nspec() const { return nelem_side1 + nelem_side2; }
  int get_npgeo() const {
    return (nelem_side1 + 1) * 2 + (nelem_side2 + 1) * 2;
  }
};

class INTERFACIAL_ASSEMBLY_FIXTURE
    : public ::testing::Test,
      public std::vector<std::pair<
          interfacial_assembly_config,
          specfem::assembly::assembly<specfem::dimension::type::dim2> > > {
protected:
  INTERFACIAL_ASSEMBLY_FIXTURE();
};

} // namespace specfem::testing
