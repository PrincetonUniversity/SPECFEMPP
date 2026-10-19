#include "specfem/assembly/sources.hpp"
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/assembly/sources/impl/locate_sources.tpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/coordinate_conversion.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/geocentric_projection.hpp"
#include "specfem/enums.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

// ── Source directional-quantity resolution (dim3) ───────────────────────────
//
// A moment tensor or force read in its local geographic frame is rotated into
// the Cartesian mesh frame here, once the source position is resolved. The
// transform is selected from the mesh projection, exactly as to_cartesian
// selects the position transform: a globe rotates by the local spherical frame
// at the source colatitude/longitude; a regional (UTM or Cartesian) mesh
// applies the flat East-North-Up relabel. A quantity already read in the
// Cartesian frame is left untouched.
namespace specfem::assembly::sources_impl {

void resolve_source_frame(
    specfem::sources::source<specfem::element::dimension_tag::dim3> &source,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

} // namespace specfem::assembly::sources_impl

void specfem::assembly::sources_impl::resolve_source_frame(
    specfem::sources::source<specfem::element::dimension_tag::dim3> &source,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  // On a globe the local frame rotates with position: recover the source
  // colatitude/longitude from its resolved (planet-centred) Cartesian position.
  // A non-globe mesh leaves the angles empty and takes the flat relabel.
  std::optional<std::pair<double, double>> angles;
  if (std::holds_alternative<specfem::assembly::mesh_impl::globe_projection>(
          mesh.projection.value)) {
    const auto global = source.get_global_coordinates();
    const auto geocentric = specfem::coordinate_systems::to<
        specfem::coordinate_systems::geocentric_coordinates>(
        specfem::coordinate_systems::cartesian_coordinates<
            specfem::element::dimension_tag::dim3>(global.x, global.y,
                                                   global.z));
    angles = std::make_pair(geocentric.theta, geocentric.phi);
  }

  if (auto *mt = dynamic_cast<specfem::sources::moment_tensor<
          specfem::element::dimension_tag::dim3> *>(&source)) {
    if (mt->moment().is_resolved())
      return;
    const auto &geographic = std::get<specfem::sources::GeographicMomentTensor>(
        mt->moment().as_read());
    mt->set_cartesian_components(
        angles ? specfem::sources::rotate_moment_tensor_geographic_to_cartesian(
                     angles->first, angles->second, geographic)
               : specfem::sources::geographic_to_enu(geographic));
  } else if (auto *f = dynamic_cast<specfem::sources::force<
                 specfem::element::dimension_tag::dim3> *>(&source)) {
    if (f->force_quantity().is_resolved())
      return;
    const auto &geographic = std::get<specfem::sources::GeographicForce>(
        f->force_quantity().as_read());
    f->set_cartesian_components(
        angles ? specfem::sources::rotate_force_geographic_to_cartesian(
                     angles->first, angles->second, geographic)
               : specfem::sources::geographic_to_enu(geographic));
  }
}

// ── Constructor template definition ─────────────────────────────────────────

template <specfem::element::dimension_tag DimensionTag>
specfem::assembly::sources<DimensionTag>::sources(
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
        &sources,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const type_real t0, const type_real dt, const int nsteps)
    : timestep(0), nspec(mesh.nspec),
      element_indices("specfem::sources::elements", sources.size()),
      h_element_indices(Kokkos::create_mirror_view(element_indices)),
      source_indices("specfem::sources::indices", sources.size()),
      h_source_indices(Kokkos::create_mirror_view(source_indices)),
      medium_types("specfem::sources::medium_types", sources.size()),
      h_medium_types(Kokkos::create_mirror_view(medium_types)),
      property_types("specfem::sources::property_types", sources.size()),
      h_property_types(Kokkos::create_mirror_view(property_types)),
      boundary_types("specfem::sources::boundary_types", sources.size()),
      h_boundary_types(Kokkos::create_mirror_view(boundary_types)),
      wavefield_types("specfem::sources::wavefield_types", sources.size()),
      h_wavefield_types(Kokkos::create_mirror_view(wavefield_types)) {

  Kokkos::deep_copy(h_element_indices, -1);
  Kokkos::deep_copy(h_source_indices, -1);

  int nsources = 0;
  int nsource_indices = 0;

  // Resolve any generic coordinates to global coordinates using mesh context.
  // The conversion (UTM for a regional mesh, spherical for a globe mesh,
  // pass-through otherwise) is selected from the mesh model. Sources
  // constructed with (x,y,z) directly have no read_coordinates_ set and keep
  // their global coordinates unchanged.
  for (auto &source : sources) {
    if (const auto *coords = source->get_read_coordinates())
      source->set_global_coordinates(
          specfem::assembly::to_cartesian(*coords, mesh));
    // Rotate a moment tensor / force read in its local geographic frame into
    // the Cartesian mesh frame now that the position is resolved. Only dim3
    // carries a resolvable local frame; dim2 sources are already Cartesian.
    if constexpr (DimensionTag == specfem::element::dimension_tag::dim3)
      specfem::assembly::sources_impl::resolve_source_frame(*source, mesh);
  }

  // Locate all sources in the mesh and set their local coordinates,
  // global element index, and medium that the source is located in
  specfem::assembly::sources_impl::locate_sources(element_types, mesh, sources);

  // Create vector of MPI slice indices for each source (host memory)
  source_partition_index_.resize(sources.size());
  for (int i = 0; i < static_cast<int>(sources.size()); ++i)
    source_partition_index_[i] = sources[i]->get_partition_index();

  // Initialize source_by_medium using TypedStorage initializer
  source_by_medium = { [&]<typename TagsType>()
                           -> SourceMediumTemplateType<TagsType> {
    constexpr auto dimension_tag = TagsType::dimension_tag;
    constexpr auto medium_tag = TagsType::medium_tag;
    auto [sorted_sources, source_indices] =
        specfem::assembly::sources_impl::sort_sources_per_medium<dimension_tag,
                                                                 medium_tag>(
            sources, element_types, mesh);

    nsources += sorted_sources.size();
    nsource_indices += source_indices.size();

    for (int isource = 0; isource < (int)sorted_sources.size(); isource++) {
      const auto &source = sorted_sources[isource];
      const auto lcoord = source->get_local_coordinates();

      int ispec = lcoord.ispec;
      const int global_isource = source_indices[isource];

      h_element_indices(global_isource) = ispec;
      h_source_indices(global_isource) = isource;
      assert(element_types.get_medium_tag(ispec) == medium_tag);
      h_medium_types(global_isource) = medium_tag;
      h_property_types(global_isource) = element_types.get_property_tag(ispec);
      h_boundary_types(global_isource) = element_types.get_boundary_tag(ispec);
      h_wavefield_types(global_isource) = source->get_wavefield_type();
    }

    return SourceMediumTemplateType<TagsType>(
        sorted_sources, mesh, jacobian_matrix, element_types, t0, dt, nsteps);
  } };

  // Count sources owned by this rank (ispec >= 0 after locate_sources).
  // In MPI builds each rank owns a subset of the global source list; in
  // serial builds every source is local, so this equals sources.size().
  int local_source_count = 0;
  for (const auto &src : sources) {
    if (src->get_local_coordinates().ispec >= 0) {
      local_source_count++;
    }
  }

  if (nsources != local_source_count) {
    std::cout << "nsources: " << nsources << std::endl;
    std::cout << "local_source_count: " << local_source_count << std::endl;
    throw std::runtime_error("Not all local sources were assigned or sources "
                             "are assigned multiple times");
  }

  auto make_source_initializer = [&](std::string label_prefix,
                                     auto index_selector, auto... tag_views) {
    return [&, label_prefix, index_selector,
            tag_views...]<typename TagsType>() -> HostIndexViewType {
      std::vector<int> matching_indices;
      matching_indices.reserve(nsources);
      for (int isource = 0; isource < (int)h_element_indices.extent(0);
           ++isource)
        if (h_element_indices(isource) >= 0 &&
            TagsType{}.has(tag_views(isource)...))
          matching_indices.emplace_back(index_selector(isource));
      HostIndexViewType host_view(label_prefix + TagsType::name(),
                                  matching_indices.size());
      for (int i = 0; i < (int)matching_indices.size(); ++i)
        host_view(i) = matching_indices[i];
      return host_view;
    };
  };

  h_source_element_by_combination = { make_source_initializer(
      "source_element_by_combination_",
      [&](int isource) { return h_element_indices(isource); }, h_medium_types,
      h_property_types, h_boundary_types, h_wavefield_types) };

  h_source_source_by_combination = { make_source_initializer(
      "source_source_by_combination_",
      [&](int isource) { return h_source_indices(isource); }, h_medium_types,
      h_property_types, h_boundary_types, h_wavefield_types) };

  source_element_by_combination =
      specfem::tag_dispatch::create_mirror_storage_and_copy(
          Kokkos::DefaultExecutionSpace{}, h_source_element_by_combination);
  source_source_by_combination =
      specfem::tag_dispatch::create_mirror_storage_and_copy(
          Kokkos::DefaultExecutionSpace{}, h_source_source_by_combination);

  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);
  Kokkos::deep_copy(property_types, h_property_types);
  Kokkos::deep_copy(boundary_types, h_boundary_types);
  Kokkos::deep_copy(element_indices, h_element_indices);
  Kokkos::deep_copy(source_indices, h_source_indices);
}

// ── get_sources_on_host / get_sources_on_device template definitions ─────────

template <specfem::element::dimension_tag DimensionTag>
std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>>
specfem::assembly::sources<DimensionTag>::get_sources_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::simulation::field_type wavefield) const {
  return std::make_tuple(h_source_element_by_combination.get(
                             medium, property, boundary, wavefield),
                         h_source_source_by_combination.get(
                             medium, property, boundary, wavefield));
}

template <specfem::element::dimension_tag DimensionTag>
std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace>>
specfem::assembly::sources<DimensionTag>::get_sources_on_device(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::simulation::field_type wavefield) const {
  return std::make_tuple(
      source_element_by_combination.get(medium, property, boundary, wavefield),
      source_source_by_combination.get(medium, property, boundary, wavefield));
}

// ── Explicit class instantiations ────────────────────────────────────────────

template class specfem::assembly::sources<
    specfem::element::dimension_tag::dim2>;

template class specfem::assembly::sources<
    specfem::element::dimension_tag::dim3>;
