#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/globe/radial_flags.hpp"
#include "specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp"
#include <type_traits>

namespace specfem::assembly::impl {

/** @brief Region and radial-zone metadata required by globe attenuation. */
struct AttenuationElementContext {
  specfem::element::region_tag region =
      specfem::element::region_tag::crust_mantle;
  int idoubling = static_cast<int>(specfem::globe::radial_flag::mantle_normal);
  /// Cartesian (non-globe) mesh: z is the radial direction, so the shift is
  /// applied without rotation and region/idoubling are ignored.
  bool cartesian = false;
  /** @brief Whether the element is a nonphysical central-cube placeholder. */
  bool fictitious() const {
    return !cartesian &&
           idoubling ==
               static_cast<int>(specfem::globe::radial_flag::fictitious_cube);
  }
  /** @brief Whether the shift needs the element's spherical coordinates. */
  bool needs_spherical_coordinates() const {
    return !cartesian && !fictitious() &&
           region != specfem::element::region_tag::inner_core;
  }
};

/**
 * @brief Attenuation context read from an assembly's element types.
 *
 * @tparam ElementTypes Assembly element types, exposing
 * has_element_context(), get_region_tag() and idoubling().
 */
template <typename ElementTypes> struct AssemblyAttenuationContext {
  const ElementTypes &element_types; ///< Element types of the assembly

  /**
   * @brief Return the metadata of a global element index.
   * @param ispec Global element index.
   * @return The element's region and radial zone, or Cartesian metadata when
   * the mesh carries no globe element context.
   */
  AttenuationElementContext operator()(const int ispec) const {
    if (!element_types.has_element_context()) {
      AttenuationElementContext context;
      context.cartesian = true;
      return context;
    }
    return { element_types.get_region_tag(ispec),
             element_types.idoubling(ispec) };
  }
};

template <typename ElementTypes>
AssemblyAttenuationContext(const ElementTypes &)
    -> AssemblyAttenuationContext<ElementTypes>;

/**
 * @brief Shift the shear moduli of a tensor given in a radial frame (third
 *        axis radial) while preserving Love's eta.
 *
 * Implements the Love-parameter update of globe's prepare_attenuation.f90
 * (reference-to-relaxed scale minus one) and prepare_elastic_elements.F90
 * (one_minus_sum_beta minus one), without the frame rotation. Cartesian
 * meshes use it directly with z as the radial axis.
 *
 * @tparam StorageType Runtime storage precision (defaults to ValueType).
 * @tparam Frame Coordinate frame of the tensor.
 * @tparam ValueType Scalar coefficient type.
 * @param radial Stiffness tensor whose third axis is radial.
 * @param minus_sum_beta Relative shear modulus increment.
 * @return Shifted stiffness tensor in the same frame.
 */
template <typename StorageType = void, typename Frame, typename ValueType>
auto shift_radial_tensor(
    specfem::medium_physics::ElasticityTensor<Frame, ValueType> radial,
    const ValueType minus_sum_beta) {
  using Storage =
      std::conditional_t<std::is_void_v<StorageType>, ValueType, StorageType>;
  if (minus_sum_beta == 0) {
    return radial;
  }
  const auto a = (3 * radial(0, 0) + 3 * radial(1, 1) + 2 * radial(0, 1) +
                  4 * radial(5, 5)) /
                 8;
  const auto n =
      (radial(0, 0) + radial(1, 1) - 2 * radial(0, 1) + 4 * radial(5, 5)) / 8;
  const auto l = (radial(3, 3) + radial(4, 4)) / 2;
  const auto f = (radial(0, 2) + radial(1, 2)) / 2;
  const ValueType eta = static_cast<Storage>(f / (a - 2 * l));
  const ValueType muv = static_cast<Storage>(l * minus_sum_beta);
  const ValueType muh = static_cast<Storage>(n * minus_sum_beta);
  const auto four_thirds = static_cast<ValueType>(4.0 / 3.0);
  const auto two_thirds = static_cast<ValueType>(2.0 / 3.0);
  radial(0, 0) += four_thirds * muh;
  radial(0, 1) -= two_thirds * muh;
  radial(0, 2) += eta * (four_thirds * muh - 2 * muv);
  radial(1, 1) += four_thirds * muh;
  radial(1, 2) += eta * (four_thirds * muh - 2 * muv);
  radial(2, 2) += four_thirds * muv;
  radial(3, 3) += muv;
  radial(4, 4) += muv;
  radial(5, 5) += muh;
  return radial;
}

/**
 * @brief Shift radial shear moduli while preserving Love's eta.
 *
 * Rotates to the local radial frame, applies shift_radial_tensor() and
 * rotates back, as globe's rotate_tensor_global_to_radial /
 * rotate_tensor_radial_to_global round trip. Call once for each setup stage.
 * An exactly zero shift returns the input without rotation round-off.
 *
 * @tparam StorageType Runtime storage precision (defaults to ValueType).
 * @tparam ValueType Scalar coefficient type.
 * @param global Cartesian stiffness tensor, after Love-to-cij conversion.
 * @param theta Geocentric colatitude in radians.
 * @param phi Geocentric longitude in radians.
 * @param minus_sum_beta Relative shear modulus increment.
 * @param radial_c44 Optional final radial c44, before the return rotation.
 * @return Shifted Cartesian stiffness tensor.
 */
template <typename StorageType = void, typename ValueType>
auto shift_unrelaxed_tensor(
    const specfem::medium_physics::ElasticityTensor<
        specfem::coordinate_systems::CartesianFrame, ValueType> &global,
    const ValueType theta, const ValueType phi, const ValueType minus_sum_beta,
    ValueType *radial_c44 = nullptr) {
  if (minus_sum_beta == 0 && radial_c44 == nullptr) {
    return global;
  }
  using Cartesian = specfem::coordinate_systems::CartesianFrame;
  using Geocentric = specfem::coordinate_systems::GeocentricFrame;
  const auto radial = shift_radial_tensor<StorageType>(
      specfem::medium_physics::rotate(
          global,
          specfem::coordinate_systems::rotation_matrix<Geocentric, Cartesian>(
              theta, phi)),
      minus_sum_beta);
  if (radial_c44 != nullptr)
    *radial_c44 = radial(3, 3);
  if (minus_sum_beta == 0) {
    return global;
  }
  return specfem::medium_physics::rotate(
      radial,
      specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(theta,
                                                                          phi));
}

/**
 * @brief Shift globe's five-coefficient, unrotated hexagonal inner-core tensor.
 * @tparam StorageType Runtime coefficient precision.
 * @tparam ValueType Rotation/register precision.
 * @param input Hexagonal stiffness with the symmetry axis along Cartesian z.
 * @param increment Relative modulus increment for this setup stage.
 * @param reference_stage True for prepare_attenuation's reference-to-relaxed
 * step.
 * @return Full Voigt expansion of the updated five independent coefficients.
 */
template <typename StorageType, typename ValueType>
auto shift_inner_core_tensor(
    const specfem::medium_physics::ElasticityTensor<
        specfem::coordinate_systems::CartesianFrame, ValueType> &input,
    const ValueType increment, const bool reference_stage) {
  using Tensor = specfem::medium_physics::ElasticityTensor<
      specfem::coordinate_systems::CartesianFrame, ValueType>;
  if (increment == 0)
    return input;
  Tensor result{};
  const ValueType mul = reference_stage
                            ? input(3, 3)
                            : static_cast<StorageType>(input(3, 3) * increment);
  const ValueType delta = reference_stage ? increment : 1;
  const auto update = [&](int row, int column, ValueType coefficient) {
    const ValueType correction = coefficient * delta * mul;
    return static_cast<ValueType>(static_cast<StorageType>(
        input(row, column) +
        (reference_stage ? static_cast<StorageType>(correction) : correction)));
  };
  result(0, 0) = result(1, 1) = update(0, 0, static_cast<ValueType>(4.0 / 3.0));
  result(0, 1) = update(0, 1, static_cast<ValueType>(-2.0 / 3.0));
  result(0, 2) = result(1, 2) =
      update(0, 2, static_cast<ValueType>(-2.0 / 3.0));
  result(2, 2) = update(2, 2, static_cast<ValueType>(4.0 / 3.0));
  result(3, 3) = result(4, 4) = update(3, 3, 1);
  result(5, 5) = static_cast<StorageType>((result(0, 0) - result(0, 1)) / 2);
  return result;
}

} // namespace specfem::assembly::impl
