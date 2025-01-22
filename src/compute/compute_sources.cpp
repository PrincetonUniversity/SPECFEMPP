#include "algorithms/interface.hpp"
#include "compute/sources/source_medium.hpp"
#include "compute/sources/source_medium.tpp"
#include "compute/sources/sources.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

// Forward declarations
namespace {
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
std::vector<std::shared_ptr<specfem::sources::source> > sort_sources_per_medium(
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const specfem::compute::element_types &element_types,
    const specfem::compute::mesh &mesh) {

  std::vector<std::shared_ptr<specfem::sources::source> > sorted_sources;

  for (int isource = 0; isource < sources.size(); isource++) {
    const auto &source = sources[isource];
    const type_real x = source->get_x();
    const type_real z = source->get_z();
    const specfem::point::global_coordinates<DimensionTag> coord(x, z);
    const auto lcoord = specfem::algorithms::locate_point(coord, mesh);
    if (element_types.get_medium_tag(lcoord.ispec) == MediumTag) {
      sorted_sources.push_back(source);
    }
  }

  return sorted_sources;
}
} // namespace

template class specfem::compute::impl::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>;

template class specfem::compute::impl::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>;

specfem::compute::sources::sources(
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::element_types &element_types, const type_real t0,
    const type_real dt, const int nsteps)
    : timestep(0), nspec(mesh.nspec),
      source_domain_index_mapping(
          "specfem::sources::source_domain_index_mapping", nspec),
      h_source_domain_index_mapping(
          Kokkos::create_mirror_view(source_domain_index_mapping)),
      medium_types("specfem::sources::medium_types", nspec),
      h_medium_types(Kokkos::create_mirror_view(medium_types)),
      property_types("specfem::sources::property_types", nspec),
      h_property_types(Kokkos::create_mirror_view(property_types)),
      boundary_types("specfem::sources::boundary_types", nspec),
      h_boundary_types(Kokkos::create_mirror_view(boundary_types)),
      wavefield_types("specfem::sources::wavefield_types", nspec),
      h_wavefield_types(Kokkos::create_mirror_view(wavefield_types)) {

  for (int ispec = 0; ispec < nspec; ispec++) {
    h_source_domain_index_mapping(ispec) = -1;
  }

#define SORT_SOURCES_PER_MEDIUM(DIMENSION_TAG, MEDIUM_TAG)                     \
  auto CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),                   \
                            GET_NAME(MEDIUM_TAG)) =                            \
      sort_sources_per_medium<GET_TAG(DIMENSION_TAG), GET_TAG(MEDIUM_TAG)>(    \
          sources, element_types, mesh);

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      SORT_SOURCES_PER_MEDIUM,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef SORT_SOURCES_PER_MEDIUM

  int nsources = 0;

#define COUNT_SOURCES(DIMENSION_TAG, MEDIUM_TAG)                               \
  nsources += CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),            \
                                   GET_NAME(MEDIUM_TAG))                       \
                  .size();

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      COUNT_SOURCES,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef COUNT_SOURCES

  if (nsources != sources.size()) {
    throw std::runtime_error(
        "Not all sources were assigned or sources are assigned multiple times");
  }

#define ASSIGN_MEMBERS(DIMENSION_TAG, MEDIUM_TAG)                              \
  {                                                                            \
    auto current_sources = CREATE_VARIABLE_NAME(                               \
        source, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG));                \
    for (int isource = 0; isource < current_sources.size(); isource++) {       \
      const auto &source = current_sources[isource];                           \
      const type_real x = source->get_x();                                     \
      const type_real z = source->get_z();                                     \
      const specfem::point::global_coordinates<GET_TAG(DIMENSION_TAG)> coord(  \
          x, z);                                                               \
      const auto lcoord = specfem::algorithms::locate_point(coord, mesh);      \
      if (lcoord.ispec < 0) {                                                  \
        throw std::runtime_error("Source is outside of the domain");           \
      }                                                                        \
      if (h_source_domain_index_mapping(lcoord.ispec) >= 0) {                  \
        throw std::runtime_error(                                              \
            "Multiple sources are detected in the same element");              \
      }                                                                        \
      h_source_domain_index_mapping(lcoord.ispec) = isource;                   \
      assert(element_types.get_medium_tag(lcoord.ispec) ==                     \
             GET_TAG(MEDIUM_TAG));                                             \
      h_medium_types(lcoord.ispec) = GET_TAG(MEDIUM_TAG);                      \
      h_property_types(lcoord.ispec) =                                         \
          element_types.get_property_tag(lcoord.ispec);                        \
      h_boundary_types(lcoord.ispec) =                                         \
          element_types.get_boundary_tag(lcoord.ispec);                        \
      h_wavefield_types(lcoord.ispec) = source->get_wavefield_type();          \
    }                                                                          \
    this->CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),                \
                               GET_NAME(MEDIUM_TAG)) =                         \
        specfem::compute::impl::source_medium<GET_TAG(DIMENSION_TAG),          \
                                              GET_TAG(MEDIUM_TAG)>(            \
            current_sources, mesh, partial_derivatives, element_types, t0, dt, \
            nsteps);                                                           \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      ASSIGN_MEMBERS,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef ASSIGN_MEMBERS

#define COUNT_SOURCES_PER_ELEMENT_TYPE(DIMENSION_TAG, MEDIUM_TAG,              \
                                       PROPERTY_TAG, BOUNDARY_TAG)             \
  int CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),             \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (h_source_domain_index_mapping(ispec) >= 0) {                           \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) &&                \
          (h_wavefield_types(ispec) ==                                         \
           specfem::wavefield::simulation_field::forward)) {                   \
        CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),           \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG))                           \
        ++;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  int CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),            \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (h_source_domain_index_mapping(ispec) >= 0) {                           \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) &&                \
          (h_wavefield_types(ispec) ==                                         \
           specfem::wavefield::simulation_field::backward)) {                  \
        CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),          \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG))                           \
        ++;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  int CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),             \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (h_source_domain_index_mapping(ispec) >= 0) {                           \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) &&                \
          (h_wavefield_types(ispec) ==                                         \
           specfem::wavefield::simulation_field::adjoint)) {                   \
        CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),           \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG))                           \
        ++;                                                                    \
      }                                                                        \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      COUNT_SOURCES_PER_ELEMENT_TYPE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef COUNT_SOURCES_PER_ELEMENT_TYPE

#define ALLOCATE_SOURCES_PER_ELEMENT_TYPE(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG, BOUNDARY_TAG)          \
  this->CREATE_VARIABLE_NAME(elements_forward, GET_NAME(DIMENSION_TAG),        \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::elements_forward",                       \
          CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),         \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(elements_backward, GET_NAME(DIMENSION_TAG),       \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::elements_backward",                      \
          CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),        \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(elements_adjoint, GET_NAME(DIMENSION_TAG),        \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::elements_adjoint",                       \
          CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),         \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(h_elements_forward, GET_NAME(DIMENSION_TAG),      \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          elements_forward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),     \
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)));                    \
  this->CREATE_VARIABLE_NAME(h_elements_backward, GET_NAME(DIMENSION_TAG),     \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          elements_backward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)));                    \
  this->CREATE_VARIABLE_NAME(h_elements_adjoint, GET_NAME(DIMENSION_TAG),      \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          elements_adjoint, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),     \
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ALLOCATE_SOURCES_PER_ELEMENT_TYPE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ALLOCATE_SOURCES_PER_ELEMENT_TYPE

#define ASSIGN_SOURCES_PER_ELEMENT_TYPE(DIMENSION_TAG, MEDIUM_TAG,             \
                                        PROPERTY_TAG, BOUNDARY_TAG)            \
  int CREATE_VARIABLE_NAME(index_forward, GET_NAME(DIMENSION_TAG),             \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (h_source_domain_index_mapping(ispec) >= 0) {                           \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) &&                \
          (h_wavefield_types(ispec) ==                                         \
           specfem::wavefield::simulation_field::forward)) {                   \
        this->CREATE_VARIABLE_NAME(                                            \
            h_elements_forward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG), \
            GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))(                   \
            CREATE_VARIABLE_NAME(index_forward, GET_NAME(DIMENSION_TAG),       \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG))) = ispec;             \
        CREATE_VARIABLE_NAME(index_forward, GET_NAME(DIMENSION_TAG),           \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG))                           \
        ++;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  int CREATE_VARIABLE_NAME(index_backward, GET_NAME(DIMENSION_TAG),            \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (h_source_domain_index_mapping(ispec) >= 0) {                           \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) &&                \
          (h_wavefield_types(ispec) ==                                         \
           specfem::wavefield::simulation_field::backward)) {                  \
        this->CREATE_VARIABLE_NAME(                                            \
            h_elements_backward, GET_NAME(DIMENSION_TAG),                      \
            GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                      \
            GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(                      \
            index_backward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),     \
            GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = ispec;          \
        CREATE_VARIABLE_NAME(index_backward, GET_NAME(DIMENSION_TAG),          \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG))                           \
        ++;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  int CREATE_VARIABLE_NAME(index_adjoint, GET_NAME(DIMENSION_TAG),             \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),       \
                           GET_NAME(BOUNDARY_TAG)) = 0;                        \
  for (int ispec = 0; ispec < nspec; ispec++) {                                \
    if (h_source_domain_index_mapping(ispec) >= 0) {                           \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) &&                \
          (h_wavefield_types(ispec) ==                                         \
           specfem::wavefield::simulation_field::adjoint)) {                   \
        this->CREATE_VARIABLE_NAME(                                            \
            h_elements_adjoint, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG), \
            GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))(                   \
            CREATE_VARIABLE_NAME(index_adjoint, GET_NAME(DIMENSION_TAG),       \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG))) = ispec;             \
        CREATE_VARIABLE_NAME(index_adjoint, GET_NAME(DIMENSION_TAG),           \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG))                           \
        ++;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  Kokkos::deep_copy(                                                           \
      this->CREATE_VARIABLE_NAME(elements_forward, GET_NAME(DIMENSION_TAG),    \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)),                      \
      this->CREATE_VARIABLE_NAME(h_elements_forward, GET_NAME(DIMENSION_TAG),  \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)));                     \
  Kokkos::deep_copy(                                                           \
      this->CREATE_VARIABLE_NAME(elements_backward, GET_NAME(DIMENSION_TAG),   \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)),                      \
      this->CREATE_VARIABLE_NAME(h_elements_backward, GET_NAME(DIMENSION_TAG), \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)));                     \
  Kokkos::deep_copy(                                                           \
      this->CREATE_VARIABLE_NAME(elements_adjoint, GET_NAME(DIMENSION_TAG),    \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)),                      \
      this->CREATE_VARIABLE_NAME(h_elements_adjoint, GET_NAME(DIMENSION_TAG),  \
                                 GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), \
                                 GET_NAME(BOUNDARY_TAG)));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ASSIGN_SOURCES_PER_ELEMENT_TYPE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ASSIGN_SOURCES_PER_ELEMENT_TYPE

  Kokkos::deep_copy(source_domain_index_mapping, h_source_domain_index_mapping);
  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);
  Kokkos::deep_copy(property_types, h_property_types);
  Kokkos::deep_copy(boundary_types, h_boundary_types);
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::sources::get_elements_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::wavefield::simulation_field wavefield) const {

#define RETURN_VALUE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG)    \
  if ((wavefield == specfem::wavefield::simulation_field::forward) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return CREATE_VARIABLE_NAME(h_elements_forward, GET_NAME(DIMENSION_TAG),   \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),  \
                                GET_NAME(BOUNDARY_TAG));                       \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::backward) &&         \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return CREATE_VARIABLE_NAME(h_elements_backward, GET_NAME(DIMENSION_TAG),  \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),  \
                                GET_NAME(BOUNDARY_TAG));                       \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::adjoint) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return CREATE_VARIABLE_NAME(h_elements_adjoint, GET_NAME(DIMENSION_TAG),   \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),  \
                                GET_NAME(BOUNDARY_TAG));                       \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      RETURN_VALUE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef RETURN_VALUE
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::sources::get_elements_on_device(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::wavefield::simulation_field wavefield) const {

#define RETURN_VALUE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG)    \
  if ((wavefield == specfem::wavefield::simulation_field::forward) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return CREATE_VARIABLE_NAME(elements_forward, GET_NAME(DIMENSION_TAG),     \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),  \
                                GET_NAME(BOUNDARY_TAG));                       \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::backward) &&         \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return CREATE_VARIABLE_NAME(elements_backward, GET_NAME(DIMENSION_TAG),    \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),  \
                                GET_NAME(BOUNDARY_TAG));                       \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::adjoint) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return CREATE_VARIABLE_NAME(elements_adjoint, GET_NAME(DIMENSION_TAG),     \
                                GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),  \
                                GET_NAME(BOUNDARY_TAG));                       \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      RETURN_VALUE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef RETURN_VALUE
}
