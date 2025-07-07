#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/assembly.hpp"

namespace specfem {
namespace coupled_interface {

namespace impl {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class coupled_interface;

template <specfem::dimension::type DimensionTag>
class coupled_interface<DimensionTag, specfem::element::medium_tag::acoustic,
                        specfem::element::medium_tag::elastic_psv> {
public:
  using CoupledPointFieldType =
      specfem::point::field<DimensionTag,
                            specfem::element::medium_tag::elastic_psv, true,
                            false, false, false, false>;
  using SelfPointFieldType =
      specfem::point::field<DimensionTag,
                            specfem::element::medium_tag::acoustic, false,
                            false, true, false, false>;
};

template <specfem::dimension::type DimensionTag>
class coupled_interface<DimensionTag, specfem::element::medium_tag::elastic_psv,
                        specfem::element::medium_tag::acoustic> {
public:
  using CoupledPointFieldType =
      specfem::point::field<DimensionTag,
                            specfem::element::medium_tag::acoustic, false,
                            false, true, false, false>;
  using SelfPointFieldType =
      specfem::point::field<DimensionTag,
                            specfem::element::medium_tag::elastic_psv, false,
                            false, true, false, false>;
};

} // namespace impl

/**
 * @brief Compute kernels to compute the coupling terms between two domains.
 *
 * @tparam WavefieldType Wavefield type on which the coupling is computed.
 * @tparam DimensionTag Dimension of the element on which the coupling is
 * computed.
 * @tparam SelfMedium Medium type of the primary domain.
 * @tparam CoupledMedium Medium type of the coupled domain.
 */
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class coupled_interface {
private:
  using CoupledPointFieldType = typename impl::coupled_interface<
      DimensionTag, SelfMedium,
      CoupledMedium>::CoupledPointFieldType; ///< Point field type of the
                                             ///< coupled domain.

  using SelfPointFieldType = typename impl::coupled_interface<
      DimensionTag, SelfMedium,
      CoupledMedium>::SelfPointFieldType; ///< Point field type of the primary
                                          ///< domain.

public:
  constexpr static auto self_medium =
      SelfMedium; ///< Medium of the primary domain.
  constexpr static auto coupled_medium =
      CoupledMedium; ///< Medium of the coupled domain.
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the element.
  constexpr static auto wavefield = WavefieldType; ///< Wavefield type.

  static_assert(SelfMedium != CoupledMedium,
                "Error: self_medium cannot be equal to coupled_medium");

  static_assert(((SelfMedium == specfem::element::medium_tag::acoustic &&
                  CoupledMedium == specfem::element::medium_tag::elastic_psv) ||
                 (SelfMedium == specfem::element::medium_tag::elastic_psv &&
                  CoupledMedium == specfem::element::medium_tag::acoustic)),
                "Only acoustic-elastic coupling is supported at the moment.");

  /**
   * @name Constructor
   */
  ///@{
  /**
   * @brief Construct a new coupled interface object.
   *
   * @param assembly Assembly object containing the mesh information.
   */
  coupled_interface(const specfem::assembly::assembly &assembly);
  ///@}

  /**
   * @name Compute coupling
   */
  void compute_coupling();

private:
  int nedges;  ///< Number of edges in the interface.
  int npoints; ///< Number of quadrature points in the interface.
  specfem::assembly::interface_container<dimension_tag, SelfMedium,
                                         CoupledMedium>
      interface_data; ///< Struct containing the coupling information.
  specfem::assembly::simulation_field<WavefieldType> field; ///< Wavefield
                                                            ///< object.
};
} // namespace coupled_interface
} // namespace specfem
