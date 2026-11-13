#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/globe/radial_flags.hpp"
#include "specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp"
#include <cmath>
#include <stdexcept>
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
 * @throws std::runtime_error if a nonzero shift has undefined or nonfinite
 * Love's eta.
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
  if (a - 2 * l == 0) {
    throw std::runtime_error(
        "Anisotropic attenuation requires A - 2L != 0 to preserve Love's "
        "eta");
  }
  const ValueType eta = static_cast<Storage>(f / (a - 2 * l));
  if (!std::isfinite(eta)) {
    throw std::runtime_error(
        "Anisotropic attenuation requires finite Love's eta = F / (A - "
        "2L)");
  }
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

/**
 * @brief Frame in which an anisotropic GLL point's shear moduli are shifted.
 */
struct RadialFrame {
  /** @brief How the radial direction is obtained. */
  enum class Kind {
    cartesian,  ///< Cartesian mesh: z is radial, no rotation.
    geocentric, ///< Globe mantle: rotate by (theta, phi).
    inner_core  ///< Globe inner core: five-coefficient hexagonal update.
  };
  Kind kind = Kind::cartesian; ///< Frame kind
  double theta = 0;            ///< Geocentric colatitude (geocentric only)
  double phi = 0;              ///< Geocentric longitude (geocentric only)

  /**
   * @brief Frame of a GLL point.
   * @param context Element metadata
   * @param theta Geocentric colatitude, read only for geocentric frames
   * @param phi Geocentric longitude, read only for geocentric frames
   * @return The point's frame
   */
  static RadialFrame of(const AttenuationElementContext &context,
                        const double theta, const double phi) {
    if (context.cartesian)
      return {};
    if (context.region == specfem::element::region_tag::inner_core)
      return { Kind::inner_core };
    return { Kind::geocentric, theta, phi };
  }
};

/**
 * @brief One setup stage of the shear shift in a point's radial frame.
 *
 * @tparam StorageType Runtime storage precision.
 * @param frame Radial frame of the point.
 * @param tensor Cartesian stiffness tensor.
 * @param increment Relative shear modulus increment for this stage.
 * @param reference_stage True for the reference-to-relaxed stage.
 * @param radial_c44 Optional output: radial c44 after the stage.
 * @return Shifted Cartesian stiffness tensor.
 */
template <typename StorageType>
specfem::medium_physics::ElasticityTensor<
    specfem::coordinate_systems::CartesianFrame, double>
shift_shear_stage(
    const RadialFrame &frame,
    const specfem::medium_physics::ElasticityTensor<
        specfem::coordinate_systems::CartesianFrame, double> &tensor,
    const double increment, const bool reference_stage,
    double *radial_c44 = nullptr) {
  if (frame.kind == RadialFrame::Kind::geocentric)
    return shift_unrelaxed_tensor<StorageType>(tensor, frame.theta, frame.phi,
                                               increment, radial_c44);
  const auto shifted =
      frame.kind == RadialFrame::Kind::inner_core
          ? shift_inner_core_tensor<StorageType>(tensor, increment,
                                                 reference_stage)
          : shift_radial_tensor<StorageType>(tensor, increment);
  if (radial_c44 != nullptr)
    *radial_c44 = shifted(3, 3);
  return shifted;
}

/**
 * @brief Shift reference-frequency shear moduli to unrelaxed ones.
 *
 * Mirrors globe's two setup stages: reference to relaxed
 * (prepare_attenuation.f90), with the coefficients stored in StorageType in
 * between, then relaxed to unrelaxed (prepare_elastic_elements.F90). The
 * input is preserved exactly when both multipliers are one.
 *
 * @tparam StorageType Runtime storage precision.
 * @param frame Radial frame of the point.
 * @param reference Cartesian stiffness tensor at the reference frequency.
 * @param relaxed_scale Reference-to-relaxed shear multiplier.
 * @param ratio Relaxed-to-unrelaxed shear multiplier (one_minus_sum_beta).
 * @param radial_c44 Output: final radial c44, globe's muvstore.
 * @return Unrelaxed Cartesian stiffness tensor.
 */
template <typename StorageType>
specfem::medium_physics::ElasticityTensor<
    specfem::coordinate_systems::CartesianFrame, double>
shift_shear_to_unrelaxed(
    const RadialFrame &frame,
    const specfem::medium_physics::ElasticityTensor<
        specfem::coordinate_systems::CartesianFrame, double> &reference,
    const double relaxed_scale, const double ratio, double &radial_c44) {
  using Tensor = specfem::medium_physics::ElasticityTensor<
      specfem::coordinate_systems::CartesianFrame, double>;
  const double stored_scale = static_cast<StorageType>(relaxed_scale);
  const auto relaxed_double = shift_shear_stage<StorageType>(
      frame, reference, stored_scale - 1.0, true);
  double relaxed_coefficients[21];
  for (int k = 0; k < 21; ++k)
    relaxed_coefficients[k] = static_cast<StorageType>(relaxed_double[k]);
  return shift_shear_stage<StorageType>(frame, Tensor(relaxed_coefficients),
                                        ratio - 1.0, false, &radial_c44);
}

/**
 * @brief Shift the bulk modulus of a tensor by an isotropic increment.
 *
 * Adds \f$\delta\kappa=\kappa_V(s-1)\f$ to the normal-stress block
 * \f$c_{ij}, i,j\le3\f$, with \f$\kappa_V\f$ the Voigt bulk modulus of
 * @p reference. The increment is an isotropic tensor, so it is
 * frame-independent, and for isotropic input it reproduces
 * \f$\kappa\leftarrow s\,\kappa\f$.
 *
 * @param tensor Tensor to shift in place.
 * @param reference Tensor whose Voigt bulk modulus is scaled.
 * @param scale Reference-to-unrelaxed bulk multiplier \f$s\f$.
 * @return Unrelaxed bulk modulus \f$s\,\kappa_V\f$.
 */
template <typename Frame>
double shift_bulk(
    specfem::medium_physics::ElasticityTensor<Frame, double> &tensor,
    const specfem::medium_physics::ElasticityTensor<Frame, double> &reference,
    const double scale) {
  const double kappa =
      (reference(0, 0) + reference(1, 1) + reference(2, 2) +
       2 * (reference(0, 1) + reference(0, 2) + reference(1, 2))) /
      9;
  if (scale == 1)
    return kappa;
  const double increment = kappa * (scale - 1);
  for (int row = 0; row < 3; ++row)
    for (int column = row; column < 3; ++column)
      tensor(row, column) += increment;
  return kappa * scale;
}

} // namespace specfem::assembly::impl
