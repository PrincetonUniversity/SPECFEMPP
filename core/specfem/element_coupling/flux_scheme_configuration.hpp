#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/quadrature/quadrature.hpp"

#include <vector>

namespace specfem::element_coupling {

/**
 * @brief Defines the (codimension-1) mesh of the interface for integration.
 *
 * The interfacial integrals are computed through composite quadrature. The
 * composite components are specified by a mesh, constructed according to one of
 * these rules.
 */
enum class interfacial_meshing_type {
  intersections, ///< integrals defined on intersections.
  acoustic_host, ///< integrals defined by element faces on acoustic side.
  elastic_host,  ///< integrals defined by element faces on elastic side.
  self_host      ///< integrals defined by element faces on self side.
};
struct flux_scheme_configuration {
private:
  // ====== acoustic-elastic ======
  specfem::element_coupling::flux_scheme_tag flux_scheme_tag;
  specfem::element_coupling::interfacial_meshing_type interfacial_meshing_type;
  specfem::quadrature::quadrature interfacial_quadrature;
  bool was_quadrature_set;
  // ==== end acoustic-elastic ====

  std::vector<type_real> scheme_parameters;

public:
  flux_scheme_configuration()
      : flux_scheme_tag(specfem::element_coupling::flux_scheme_tag::natural),
        was_quadrature_set(false),
        interfacial_meshing_type(
            specfem::
                element_coupling:: // this may differ from defaults in
                                   // core/specfem/runtime_configuration/flux_schemes.hpp
            interfacial_meshing_type::intersections) {}

  /**
   * @brief Get the flux scheme tag for a given intersection. At the moment, the
   * flux scheme tag is global, so only one tag will ever be returned per
   * simulation.
   *
   * @return specfem::element_coupling::flux_scheme_tag the flux scheme
   * corresponding to the intersection.
   */
  specfem::element_coupling::flux_scheme_tag get_flux_scheme_tag() const {
    return flux_scheme_tag;
  }

  /**
   * @brief Sets the flux scheme tag for the acoustic-elastic interface.
   */
  void set_flux_scheme_tag(
      specfem::element_coupling::flux_scheme_tag flux_scheme_tag) {
    this->flux_scheme_tag = flux_scheme_tag;
  }

  /**
   * @brief Get the quadrature object corresponding to interfacial integration.
   *
   * @return specfem::quadrature::quadrature the quadrature object for the
   * interface.
   */
  specfem::quadrature::quadrature get_interfacial_quadrature() const {
    return interfacial_quadrature;
  }

  /**
   * @brief Returns whether or not the interfacial_quadrature object was set. If
   * False, quadrature Views may not be initialized.
   */
  bool was_interfacial_quadrature_set() const { return was_quadrature_set; }

  /**
   * @brief Sets the quadrature object corresponding to interfacial integration.
   *
   */
  void set_interfacial_quadrature(
      specfem::quadrature::quadrature interfacial_quadrature) {
    this->interfacial_quadrature = interfacial_quadrature;
    was_quadrature_set = true;
  }

  /**
   * @brief Get the interfacial meshing type for the acoustic-elastic interface.
   *
   * @return specfem::element_coupling::interfacial_meshing_type the meshing
   * type
   */
  specfem::element_coupling::interfacial_meshing_type
  get_interfacial_meshing_type() const {
    return interfacial_meshing_type;
  }
  /**
   * @brief Set the interfacial meshing type for the acoustic-elastic interface
   *
   * @param interfacial_meshing_type the value to set to
   */
  void set_interfacial_meshing_type(
      specfem::element_coupling::interfacial_meshing_type
          interfacial_meshing_type) {
    this->interfacial_meshing_type = interfacial_meshing_type;
  }
};
} // namespace specfem::element_coupling
