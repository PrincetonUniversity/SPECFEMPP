#include "specfem/assembly/sources.hpp"
#include "../impl/dim2/source_medium.tpp"
#include "../impl/locate_sources.hpp"
#include "../impl/locate_sources.tpp"
#include "../impl/source_medium.hpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

template void specfem::assembly::sources_impl::locate_sources<
    specfem::element::dimension_tag::dim2>(
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &,
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::element::dimension_tag::dim2> > > &);

template class specfem::assembly::sources_impl::source_medium<
    specfem::element::dimension_tag::dim2,
    specfem::element::medium_tag::acoustic>;

template class specfem::assembly::sources_impl::source_medium<
    specfem::element::dimension_tag::dim2,
    specfem::element::medium_tag::elastic_psv>;

template class specfem::assembly::sources_impl::source_medium<
    specfem::element::dimension_tag::dim2,
    specfem::element::medium_tag::poroelastic>;

template class specfem::assembly::sources_impl::source_medium<
    specfem::element::dimension_tag::dim2,
    specfem::element::medium_tag::elastic_psv_t>;

specfem::assembly::sources<specfem::element::dimension_tag::dim2>::sources(
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::element::dimension_tag::dim2> > >
        &sources,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<
        specfem::element::dimension_tag::dim2> &jacobian_matrix,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
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

  // Here we sort the sources by the different media and create
  // a vector of sources for each medium named source_<dim>_<medium>
  // and a vector of indices of the sources in the original sources vector
  // named source_indices_<dim>_<medium>

  int nsources = 0;
  int nsource_indices = 0;

  // Locate all sources in the mesh and set their local coordinates,
  // global element index, and medium that the source is located in
  specfem::assembly::sources_impl::locate_sources(element_types, mesh, sources);

  // Initialize source_by_medium using TypedStorage initializer
  source_by_medium = decltype(
      source_by_medium)([&]<typename TagsType>() -> SourceMediumFor<TagsType> {
    constexpr auto dim_tag = TagsType::dimension_tag;
    constexpr auto med_tag = TagsType::medium_tag;
    auto [sorted_sources, source_indices] =
        specfem::assembly::sources_impl::sort_sources_per_medium<dim_tag,
                                                                 med_tag>(
            sources, element_types, mesh);

    /** For a sanity check we count the number of sources and source indices
     * for each medium and dimension
     */
    nsources += sorted_sources.size();
    nsource_indices += source_indices.size();

    /* Loops over the current source*/
    for (int isource = 0; isource < sorted_sources.size(); isource++) {
      const auto &source = sorted_sources[isource];
      const auto lcoord = source->get_local_coordinates();

      int ispec = lcoord.ispec;
      const int global_isource = source_indices[isource];

      /* setting local source to global element mapping */
      h_element_indices(global_isource) = ispec;
      assert(element_types.get_medium_tag(ispec) == med_tag);
      h_medium_types(global_isource) = med_tag;
      h_property_types(global_isource) = element_types.get_property_tag(ispec);
      h_attenuation_types(global_isource) =
          element_types.get_attenuation_tag(ispec);
      h_boundary_types(global_isource) = element_types.get_boundary_tag(ispec);
      h_wavefield_types(global_isource) = source->get_wavefield_type();
    }

    return SourceMediumFor<TagsType>(sorted_sources, mesh, jacobian_matrix,
                                     element_types, t0, dt, nsteps);
  });

  // if the number of sources is not equal to the number of sources
  if (nsources != sources.size()) {
    std::cout << "nsources: " << nsources << std::endl;
    std::cout << "sources.size(): " << sources.size() << std::endl;
    throw std::runtime_error(
        "Not all sources were assigned or sources are assigned multiple times");
  }

  // Initialize h_source_index_by_combination and source_index_by_combination
  // using Storage initializer, keyed by (dim, medium, property, attenuation,
  // boundary, wavefield)
  h_source_index_by_combination = decltype(h_source_index_by_combination)(
      [&]<typename TagsType>() -> HostIndexPairType {
        constexpr auto med_tag = TagsType::medium_tag;
        constexpr auto prop_tag = TagsType::property_tag;
        constexpr auto atten_tag = TagsType::attenuation_tag;
        constexpr auto bnd_tag = TagsType::boundary_tag;
        constexpr auto wf_tag = TagsType::wavefield_tag;

        int count = 0;
        for (int isource = 0; isource < sources.size(); isource++) {
          if (h_medium_types(isource) == med_tag &&
              h_property_types(isource) == prop_tag &&
              h_attenuation_types(isource) == atten_tag &&
              h_boundary_types(isource) == bnd_tag &&
              h_wavefield_types(isource) == wf_tag) {
            ++count;
          }
        }

        IndexViewType::HostMirror h_elem(
            "specfem::assembly::sources::element_indices", count);
        IndexViewType::HostMirror h_src(
            "specfem::assembly::sources::source_indices", count);

        int idx = 0;
        for (int isource = 0; isource < sources.size(); isource++) {
          if (h_medium_types(isource) == med_tag &&
              h_property_types(isource) == prop_tag &&
              h_attenuation_types(isource) == atten_tag &&
              h_boundary_types(isource) == bnd_tag &&
              h_wavefield_types(isource) == wf_tag) {
            h_elem(idx) = h_element_indices(isource);
            h_src(idx) = isource;
            ++idx;
          }
        }

        return { h_elem, h_src };
      });

  source_index_by_combination =
      specfem::tag_dispatch::create_mirror_storage_and_copy(
          Kokkos::DefaultExecutionSpace{}, h_source_index_by_combination);

  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);
  Kokkos::deep_copy(property_types, h_property_types);
  Kokkos::deep_copy(attenuation_types, h_attenuation_types);
  Kokkos::deep_copy(boundary_types, h_boundary_types);
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::sources<specfem::element::dimension_tag::dim2>::
    get_sources_on_host(const specfem::element::medium_tag medium,
                        const specfem::element::property_tag property,
                        const specfem::element::attenuation_tag attenuation,
                        const specfem::element::boundary_tag boundary,
                        const specfem::simulation::field_type wavefield) const {
  const auto &[h_elem, h_src] = h_source_index_by_combination.get(
      medium, property, attenuation, boundary, wavefield);
  return std::make_tuple(h_elem, h_src);
}

// This function is crucial for the computing the source contribution
// to the wavefield. It returns the global indices of the relevant elements
// and the source indices for the wavefield type.
std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::assembly::sources<specfem::element::dimension_tag::dim2>::
    get_sources_on_device(
        const specfem::element::medium_tag medium,
        const specfem::element::property_tag property,
        const specfem::element::attenuation_tag attenuation,
        const specfem::element::boundary_tag boundary,
        const specfem::simulation::field_type wavefield) const {
  const auto &[d_elem, d_src] = source_index_by_combination.get(
      medium, property, attenuation, boundary, wavefield);
  return std::make_tuple(d_elem, d_src);
}
