#pragma once

#include "enumerations/connections.hpp"
#include "enumerations/coupled_interface.hpp"
#include "specfem/element.hpp"
#include "specfem/macros.hpp"
#include <array>
#include <boost/preprocessor.hpp>
#include <tuple>

namespace specfem::interface {
/**
 * @brief A constexpr function to generate a list of edges with interfaces
 * within the simulation.
 *
 * This function uses @ref EDGES to generate a list of edges automatically.
 *
 * @return constexpr auto list of edges
 */
template <specfem::element::dimension_tag DimensionTag> constexpr auto edges();

/**
 * @brief 2D specialization of the edges function
 *
 * @return constexpr auto list of edges for 2D
 */
template <> constexpr auto edges<specfem::element::dimension_tag::dim2>() {
  constexpr int total_edges = BOOST_PP_SEQ_SIZE(EDGES);
  constexpr std::array<
      std::tuple<specfem::element::dimension_tag, specfem::connections::type,
                 specfem::interface::interface_tag,
                 specfem::element::boundary_tag>,
      total_edges>
      edges{ _MAKE_CONSTEXPR_ARRAY(EDGES) };
  return edges;
}
} // namespace specfem::interface
