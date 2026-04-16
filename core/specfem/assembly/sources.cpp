#include "specfem/assembly/sources.hpp"
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/assembly/sources/impl/locate_sources.tpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

// ── Constructor template definition ─────────────────────────────────────────

template <specfem::element::dimension_tag DimensionTag>
specfem::assembly::sources<DimensionTag>::sources(
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
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
      attenuation_types("specfem::sources::attenuation_types", sources.size()),
      h_attenuation_types(Kokkos::create_mirror_view(attenuation_types)),
      boundary_types("specfem::sources::boundary_types", sources.size()),
      h_boundary_types(Kokkos::create_mirror_view(boundary_types)),
      wavefield_types("specfem::sources::wavefield_types", sources.size()),
      h_wavefield_types(Kokkos::create_mirror_view(wavefield_types)) {

  int nsources = 0;
  int nsource_indices = 0;

  // Locate all sources in the mesh and set their local coordinates,
  // global element index, and medium that the source is located in
  specfem::assembly::sources_impl::locate_sources(element_types, mesh, sources);

  // Initialize source_by_medium using TypedStorage initializer
  source_by_medium = decltype(source_by_medium)(
      [&]<typename TagsType>() -> SourceMediumTemplateType<TagsType> {
        constexpr auto dim_tag = TagsType::dimension_tag;
        constexpr auto med_tag = TagsType::medium_tag;
        auto [sorted_sources, source_indices] =
            specfem::assembly::sources_impl::sort_sources_per_medium<dim_tag,
                                                                     med_tag>(
                sources, element_types, mesh);

        nsources += sorted_sources.size();
        nsource_indices += source_indices.size();

        for (int isource = 0; isource < (int)sorted_sources.size(); isource++) {
          const auto &source = sorted_sources[isource];
          const auto lcoord = source->get_local_coordinates();

          int ispec = lcoord.ispec;
          const int global_isource = source_indices[isource];

          h_element_indices(global_isource) = ispec;
          assert(element_types.get_medium_tag(ispec) == med_tag);
          h_medium_types(global_isource) = med_tag;
          h_property_types(global_isource) =
              element_types.get_property_tag(ispec);
          h_attenuation_types(global_isource) =
              element_types.get_attenuation_tag(ispec);
          h_boundary_types(global_isource) =
              element_types.get_boundary_tag(ispec);
          h_wavefield_types(global_isource) = source->get_wavefield_type();
        }

        return SourceMediumTemplateType<TagsType>(
            sorted_sources, mesh, jacobian_matrix, element_types, t0, dt,
            nsteps);
      });

  if (nsources != (int)sources.size()) {
    std::cout << "nsources: " << nsources << std::endl;
    std::cout << "sources.size(): " << sources.size() << std::endl;
    throw std::runtime_error(
        "Not all sources were assigned or sources are assigned multiple times");
  }

  int nsources_total = (int)sources.size();

  auto make_source_initializer = [&](std::string label_prefix,
                                     auto index_selector, auto... tag_views) {
    return [&, label_prefix, index_selector,
            tag_views...]<typename TagsType>() -> HostIndexViewType {
      std::vector<int> matching_indices;
      matching_indices.reserve(nsources_total);
      for (int isource = 0; isource < nsources_total; ++isource)
        if (TagsType{}.has(tag_views(isource)...))
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
      h_property_types, h_attenuation_types, h_boundary_types,
      h_wavefield_types) };

  h_source_source_by_combination = { make_source_initializer(
      "source_source_by_combination_", [&](int isource) { return isource; },
      h_medium_types, h_property_types, h_attenuation_types, h_boundary_types,
      h_wavefield_types) };

  source_element_by_combination =
      specfem::tag_dispatch::create_mirror_storage_and_copy(
          Kokkos::DefaultExecutionSpace{}, h_source_element_by_combination);
  source_source_by_combination =
      specfem::tag_dispatch::create_mirror_storage_and_copy(
          Kokkos::DefaultExecutionSpace{}, h_source_source_by_combination);

  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);
  Kokkos::deep_copy(property_types, h_property_types);
  Kokkos::deep_copy(attenuation_types, h_attenuation_types);
  Kokkos::deep_copy(boundary_types, h_boundary_types);
}

// ── get_sources_on_host / get_sources_on_device template definitions ─────────

template <specfem::element::dimension_tag DimensionTag>
std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::sources<DimensionTag>::get_sources_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::attenuation_tag attenuation,
    const specfem::element::boundary_tag boundary,
    const specfem::simulation::field_type wavefield) const {
  return std::make_tuple(
      h_source_element_by_combination.get(medium, property, attenuation,
                                          boundary, wavefield),
      h_source_source_by_combination.get(medium, property, attenuation,
                                         boundary, wavefield));
}

template <specfem::element::dimension_tag DimensionTag>
std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::assembly::sources<DimensionTag>::get_sources_on_device(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::attenuation_tag attenuation,
    const specfem::element::boundary_tag boundary,
    const specfem::simulation::field_type wavefield) const {
  return std::make_tuple(
      source_element_by_combination.get(medium, property, attenuation, boundary,
                                        wavefield),
      source_source_by_combination.get(medium, property, attenuation, boundary,
                                       wavefield));
}

// ── Explicit class instantiations ────────────────────────────────────────────

template class specfem::assembly::sources<
    specfem::element::dimension_tag::dim2>;

template class specfem::assembly::sources<
    specfem::element::dimension_tag::dim3>;
