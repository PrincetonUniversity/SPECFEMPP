
#include "compute/boundaries/impl/stacey.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>
#include <vector>

namespace {
bool is_on_boundary(specfem::enums::boundaries::type type, int iz, int ix,
                    int ngllz, int ngllx) {
  return (type == specfem::enums::boundaries::type::TOP && iz == ngllz - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM && iz == 0) ||
         (type == specfem::enums::boundaries::type::LEFT && ix == 0) ||
         (type == specfem::enums::boundaries::type::RIGHT && ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM_RIGHT && iz == 0 &&
          ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM_LEFT && iz == 0 &&
          ix == 0) ||
         (type == specfem::enums::boundaries::type::TOP_RIGHT &&
          iz == ngllz - 1 && ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::TOP_LEFT &&
          iz == ngllz - 1 && ix == 0);
}

std::tuple<std::array<type_real, 2>, type_real> get_boundary_edge_and_weight(
    specfem::enums::boundaries::type type,
    const std::array<type_real, 2> &weights,
    const specfem::point::partial_derivatives2<false, true>
        &point_partial_derivatives) {

  if (type == specfem::enums::boundaries::type::BOTTOM_LEFT ||
      type == specfem::enums::boundaries::type::TOP_LEFT ||
      type == specfem::enums::boundaries::type::LEFT) {
    const auto normal = point_partial_derivatives.compute_normal(
        specfem::enums::edge::type::LEFT);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[1]);
  }

  if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT ||
      type == specfem::enums::boundaries::type::TOP_RIGHT ||
      type == specfem::enums::boundaries::type::RIGHT) {
    const auto normal = point_partial_derivatives.compute_normal(
        specfem::enums::edge::type::RIGHT);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[1]);
  }

  if (type == specfem::enums::boundaries::type::TOP) {
    const auto normal = point_partial_derivatives.compute_normal(
        specfem::enums::edge::type::TOP);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[0]);
  }

  if (type == specfem::enums::boundaries::type::BOTTOM) {
    const auto normal = point_partial_derivatives.compute_normal(
        specfem::enums::edge::type::BOTTOM);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[0]);
  }

  throw std::invalid_argument("Error: Unknown boundary type");
}
} // namespace

specfem::compute::impl::boundaries::stacey::stacey(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::mesh::absorbing_boundary &stacey,
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::compute::quadrature &quadrature,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
    const Kokkos::View<specfem::element::boundary_tag_container *,
                       Kokkos::HostSpace> &element_boundary_tags) {

  // We need to make sure that boundary index mapping maps every spectral
  // element index to the corresponding index within
  // quadrature_point_boundary_tag

  // mesh.absorbing_boundary.ispec_absorbing_boundary stores the ispec for
  // every Stacey surface. At the corners of the mesh, multiple surfaces
  // belong to the same ispec. The first part of the code assigns a unique index
  // to each ispec.

  // For SIMD loads we need to ensure that there is a contiguous mapping within
  // ispec and index of boundary_index_mapping i.e. boundary_index_mapping(ispec
  // +1) - boundary_index_mapping(ispec) = 1

  // ------------------- Sort ispec_absorbing_boundary -------------------
  // There might be better way of doing this but for now I am sorting
  const int nelements = stacey.nelements;
  std::vector<int> sorted_ispec(nelements);
  std::vector<specfem::enums::boundaries::type> sorted_type(nelements);

  std::vector<std::size_t> iota(nelements);

  std::iota(iota.begin(), iota.end(), 0);

  // Sort indices based on ispec_absorbing_boundary
  std::sort(iota.begin(), iota.end(), [&](std::size_t i1, std::size_t i2) {
    return stacey.ispec(i1) < stacey.ispec(i2);
  });

  // Reorder ispec_absorbing_boundary and type
  for (int i = 0; i < nelements; ++i) {
    sorted_ispec[i] = stacey.ispec(iota[i]);
    sorted_type[i] = stacey.type(iota[i]);
  }
  // -------------------------------------------------------------------

  // ------------------- Assign boundary index mapping -------------------
  // Initialize all index mappings to -1
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // Assign boundary index mapping
  int total_indices = 0;
  for (int i = 0; i < nelements; ++i) {
    const int ispec = sorted_ispec[i];
    const int ispec_compute = mapping.mesh_to_compute(ispec);
    if (boundary_index_mapping(ispec_compute) == -1) {
      boundary_index_mapping(ispec_compute) = total_indices;
      ++total_indices;
    }
  }

  // Make sure the index mapping is contiguous
  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (ispec == 0)
      continue;

    if ((boundary_index_mapping(ispec) == -1) &&
        (boundary_index_mapping(ispec - 1) != -1)) {
      throw std::invalid_argument(
          "Error: Boundary index mapping is not contiguous");
    } else {
      continue;
    }

    if (boundary_index_mapping(ispec) !=
        boundary_index_mapping(ispec - 1) + 1) {
      throw std::invalid_argument(
          "Error: Boundary index mapping is not contiguous");
    }
  }

  // ------------------- Assign quadrature point boundary tag
  // ------------------- Assign boundary tags

  // Initialize boundary tags
  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::compute::impl::boundaries::"
                      "acoustic_free_surface::quadrature_point_boundary_tag",
                      total_indices, ngllz, ngllx);

  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  this->h_edge_weight = EdgeWeightView("specfem::compute::impl::boundaries::"
                                       "acoustic_free_surface::edge_weight",
                                       total_indices, ngllz, ngllx);

  this->h_edge_normal = EdgeNormalView("specfem::compute::impl::boundaries::"
                                       "acoustic_free_surface::edge_normal",
                                       total_indices, ngllz, ngllx, 2);

  this->h_edge_weight = Kokkos::create_mirror_view(edge_weight);
  this->h_edge_normal = Kokkos::create_mirror_view(edge_normal);

  for (int i = 0; i < nelements; ++i) {
    const int ispec = sorted_ispec[i];
    const int ispec_compute = mapping.mesh_to_compute(ispec);
    const auto type = sorted_type[i];
    const int local_index = boundary_index_mapping(ispec_compute);

    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ix = 0; ix < ngllx; ++ix) {
        if (is_on_boundary(type, iz, ix, ngllz, ngllx)) {
          this->h_quadrature_point_boundary_tag(local_index, iz, ix) +=
              specfem::element::boundary_tag::stacey;

          // Compute edge normal and edge weight
          std::array<type_real, 2> weights = { quadrature.gll.h_weights(ix),
                                               quadrature.gll.h_weights(iz) };
          specfem::point::index index(ispec_compute, iz, ix);
          specfem::point::partial_derivatives2<false, true>
              point_partial_derivatives;
          specfem::compute::load_on_host(index, partial_derivatives,
                                         point_partial_derivatives);

          auto [edge_normal, edge_weight] = get_boundary_edge_and_weight(
              type, weights, point_partial_derivatives);
          // ------------------- Assign edge normal and edge weight

          this->h_edge_weight(local_index, iz, ix) = edge_weight;
          this->h_edge_normal(local_index, iz, ix, 0) = edge_normal[0];
          this->h_edge_normal(local_index, iz, ix, 1) = edge_normal[1];
        } else {
          this->h_quadrature_point_boundary_tag(local_index, iz, ix) +=
              specfem::element::boundary_tag::none;
          this->h_edge_weight(local_index, iz, ix) = 0.0;
          this->h_edge_normal(local_index, iz, ix, 0) = 0.0;
          this->h_edge_normal(local_index, iz, ix, 1) = 0.0;
        }
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
  Kokkos::deep_copy(edge_weight, h_edge_weight);
  Kokkos::deep_copy(edge_normal, h_edge_normal);
}
