#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace coupled_interface {

namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class coupled_interface;

template <specfem::dimension::type DimensionType>
class coupled_interface<DimensionType, specfem::element::medium_tag::acoustic,
                        specfem::element::medium_tag::elastic_sv> {
public:
  using CoupledPointFieldType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::elastic_sv, true,
                            false, false, false, false>;
  using SelfPointFieldType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, false,
                            false, true, false, false>;
};

template <specfem::dimension::type DimensionType>
class coupled_interface<DimensionType, specfem::element::medium_tag::elastic_sv,
                        specfem::element::medium_tag::acoustic> {
public:
  using CoupledPointFieldType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, false,
                            false, true, false, false>;
  using SelfPointFieldType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::elastic_sv, false,
                            false, true, false, false>;
};

} // namespace impl

/**
 * @brief Compute kernels to compute the coupling terms between two domains.
 *
 * @tparam WavefieldType Wavefield type on which the coupling is computed.
 * @tparam DimensionType Dimension of the element on which the coupling is
 * computed.
 * @tparam SelfMedium Medium type of the primary domain.
 * @tparam CoupledMedium Medium type of the coupled domain.
 */
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class coupled_interface {
private:
  using CoupledPointFieldType = typename impl::coupled_interface<
      DimensionType, SelfMedium,
      CoupledMedium>::CoupledPointFieldType; ///< Point field type of the
                                             ///< coupled domain.

  using SelfPointFieldType = typename impl::coupled_interface<
      DimensionType, SelfMedium,
      CoupledMedium>::SelfPointFieldType; ///< Point field type of the primary
                                          ///< domain.

public:
  constexpr static auto self_medium =
      SelfMedium; ///< Medium of the primary domain.
  constexpr static auto coupled_medium =
      CoupledMedium; ///< Medium of the coupled domain.
  constexpr static auto dimension =
      DimensionType; ///< Dimension of the element.
  constexpr static auto wavefield = WavefieldType; ///< Wavefield type.

  static_assert(SelfMedium != CoupledMedium,
                "Error: self_medium cannot be equal to coupled_medium");

  static_assert(((SelfMedium == specfem::element::medium_tag::acoustic &&
                  CoupledMedium == specfem::element::medium_tag::elastic_sv) ||
                 (SelfMedium == specfem::element::medium_tag::elastic_sv &&
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
  coupled_interface(const specfem::compute::assembly &assembly);
  ///@}

  /**
   * @name Compute coupling
   */
  void compute_coupling();

private:
  int nedges;  ///< Number of edges in the interface.
  int npoints; ///< Number of quadrature points in the interface.
  specfem::compute::interface_container<SelfMedium, CoupledMedium>
      interface_data; ///< Struct containing the coupling information.
  specfem::compute::simulation_field<WavefieldType> field; ///< Wavefield
                                                           ///< object.
};
} // namespace coupled_interface
} // namespace specfem
