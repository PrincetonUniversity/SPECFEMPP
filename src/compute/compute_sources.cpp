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

/* THIS FUNCTION WILL HAVE TO CHANGE. IF YOU HAVE MANY SOURCES IN
 * MULTIPLE DOMAINS THE SOURCES ARE BEING LOCATED MULTIPLE TIMES
 * FOR ONE SOURCE, THIS WILL NOT HAVE AN IMPACT AT ALL, BUT
 *
 * The function runs for every material type and returns vector of sources that
 * fall into that material domain
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
std::vector<std::shared_ptr<specfem::sources::source> > sort_sources_per_medium(
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const specfem::compute::element_types &element_types,
    const specfem::compute::mesh &mesh) {

  std::vector<std::shared_ptr<specfem::sources::source> > sorted_sources;

  // Loop over all sources
  for (int isource = 0; isource < sources.size(); isource++) {

    // Get the source coordinates
    const auto &source = sources[isource];
    const type_real x = source->get_x();
    const type_real z = source->get_z();

    // Get element that the source is located in
    const specfem::point::global_coordinates<DimensionTag> coord(x, z);
    const auto lcoord = specfem::algorithms::locate_point(coord, mesh);

    // Check if the element is in currently checked medium and add to
    // the list of sources and indices if it is.
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
      elements_indices("specfem::sources::elements", sources.size()),
      h_elements_indices(Kokkos::create_mirror_view(elements)),
      source_indices("specfem::sources::indeces", sources.size()),
      h_source_indices(Kokkos::create_mirror_view(source_indices)),
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

  // THERE SHOULD BE LOCATE SOURCES HERE, AND SOURCE SHOULD BE POPULATED
  // WITH THE LOCAL COORDINATES AND THE GLOBAL ELEMENT INDEX

// Here we sort the sources by the different media and create
// a vector of sources for each medium named source_<dim>_<medium>
// and a vector of indices of the sources in the original sources vector
// named source_indices_<dim>_<medium>
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
// For a sanity check we count the number of sources and source indices
// for each medium and dimension
#define COUNT_SOURCES(DIMENSION_TAG, MEDIUM_TAG)                               \
  nsources += CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),            \
                                   GET_NAME(MEDIUM_TAG))                       \
                  .size();

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      COUNT_SOURCES,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef COUNT_SOURCES

  // if the number of sources is not equal to the number of sources
  if (nsources != sources.size()) {
    std::cout << "nsources: " << nsources << std::endl;
    std::cout << "sources.size(): " << sources.size() << std::endl;
    throw std::runtime_error(
        "Not all sources were assigned or sources are assigned multiple times");
  }

  // if the number of source indices is not equal to the number of sources
  if (nsource_indices != sources.size()) {
    std::cout << "nsource_indices: " << nsource_indices << std::endl;
    std::cout << "sources.size(): " << sources.size() << std::endl;
    throw std::runtime_error(
        "Not all sources were assigned or sources are assigned multiple times");
  }

  // Reminder we already have
  //    vector<source> current_sources =  source_<dim>_<medium>
  // and
  //    vector<int> source_indices = source_indices_<dim>_<medium>

#define ASSIGN_MEMBERS(DIMENSION_TAG, MEDIUM_TAG)                              \
  {                                                                            \
    /* Gets the sources and global indices for the current source medium */    \
    auto current_sources = CREATE_VARIABLE_NAME(                               \
        source, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG));                \
    /* Loops over the current source*/                                         \
    for (int isource = 0; isource < current_sources.size(); isource++) {       \
      const auto &source = current_sources[isource];                           \
      const type_real x = source->get_x();                                     \
      const type_real z = source->get_z();                                     \
      const specfem::point::global_coordinates<GET_TAG(DIMENSION_TAG)> coord(  \
          x, z);                                                               \
      /* Locates the source in the mesh. Should have been stored already. */   \
      const auto lcoord = specfem::algorithms::locate_point(coord, mesh);      \
      if (lcoord.ispec < 0) {                                                  \
        throw std::runtime_error("Source is outside of the domain");           \
      }                                                                        \
      int ispec = lcoord.ispec;                                                \
      /* if (h_source_domain_index_mapping(lcoord.ispec) >= 0) {               \
       *  throw std::runtime_error(                                            \
       *       "Multiple sources are detected in the same element");           \
       * }                                                                     \
       */                                                                      \
      /* source_domain index mapping will be removed */                        \
      h_source_domain_index_mapping(ispec) = isource;                          \
      /* setting local source to global element mapping */                     \
      h_elements(isource) = ispec;                                             \
      assert(element_types.get_medium_tag(ispec) == GET_TAG(MEDIUM_TAG));      \
      h_medium_types(ispec) = GET_TAG(MEDIUM_TAG);                             \
      h_property_types(ispec) = element_types.get_property_tag(ispec);         \
      h_boundary_types(ispec) = element_types.get_boundary_tag(ispec);         \
      h_wavefield_types(ispec) = source->get_wavefield_type();                 \
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

#define COUNT_SOURCES_PER_ELEMENT_TYPE(DIMENSION_TAG, MEDIUM_TAG,               \
                                       PROPERTY_TAG, BOUNDARY_TAG)              \
  /* Initialize base counts */                                                  \
  int CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),              \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),        \
                           GET_NAME(BOUNDARY_TAG)) = 0;                         \
  int CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),             \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),        \
                           GET_NAME(BOUNDARY_TAG)) = 0;                         \
  int CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),              \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),        \
                           GET_NAME(BOUNDARY_TAG)) = 0;                         \
                                                                                \
  /* Loop over the sources and count element type specific source count */      \
  for (int isource = 0; isource < sources.size(); isource++) {                  \
    int ispec = h_elements(isource);                                            \
    /* Check if the ispec belongs to base element type  */                      \
      if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
          (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
          (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) {                \
          /* Check the wavefield type and increment the count */              \
          if (h_wavefield_types(ispec) ==                                       \
              specfem::wavefield::simulation_field::forward) {                 \
              CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),     \
                                   GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),\
                                   GET_NAME(BOUNDARY_TAG))++;                   \
          } else if (h_wavefield_types(ispec) ==                               \
                     specfem::wavefield::simulation_field::backward) {         \
              CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),    \
                                   GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),\
                                   GET_NAME(BOUNDARY_TAG))++;                   \
          } else if (h_wavefield_types(ispec) ==                               \
                     specfem::wavefield::simulation_field::adjoint) {          \
              CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),     \
                                   GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),\
                                   GET_NAME(BOUNDARY_TAG))++;                   \
          }                                                                     \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      COUNT_SOURCES_PER_ELEMENT_TYPE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef COUNT_SOURCES_PER_ELEMENT_TYPE

// Allocate memory for the sources per element type
#define ALLOCATE_SOURCES_PER_ELEMENT_TYPE(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG, BOUNDARY_TAG)          \
  /* ==================================== */                                   \
  /* Allocating the element specific element_indices array */                  \
  this->CREATE_VARIABLE_NAME(element_indices_forward, GET_NAME(DIMENSION_TAG), \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::element_indices_forward",                \
          CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),         \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(element_indices_backward,                         \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      IndexViewType(                                                           \
          "specfem::compute::sources::element_indices_backward",               \
          CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),        \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(element_indices_adjoint, GET_NAME(DIMENSION_TAG), \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::element_indices_adjoint",                \
          CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),         \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(h_element_indices_forward,                        \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          element_indices_forward, GET_NAME(DIMENSION_TAG),                    \
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                        \
          GET_NAME(BOUNDARY_TAG)));                                            \
  this->CREATE_VARIABLE_NAME(h_element_indices_backward,                       \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          element_indices_backward, GET_NAME(DIMENSION_TAG),                   \
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                        \
          GET_NAME(BOUNDARY_TAG)));                                            \
  this->CREATE_VARIABLE_NAME(h_element_indices_adjoint,                        \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          element_indices_adjoint, GET_NAME(DIMENSION_TAG),                    \
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                        \
          GET_NAME(BOUNDARY_TAG)));                                            \
  /* ==================================== */                                   \
  /* Allocation the element specific source_indices arrays */                  \
  this->CREATE_VARIABLE_NAME(source_indices_forward, GET_NAME(DIMENSION_TAG),  \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::source_indices_forward",                 \
          CREATE_VARIABLE_NAME(count_forward, GET_NAME(DIMENSION_TAG),         \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(source_indices_backward, GET_NAME(DIMENSION_TAG), \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::source_indices_backward",                \
          CREATE_VARIABLE_NAME(count_backward, GET_NAME(DIMENSION_TAG),        \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(source_indices_adjoint, GET_NAME(DIMENSION_TAG),  \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)) =                         \
      IndexViewType(                                                           \
          "specfem::compute::sources::source_indices_adjoint",                 \
          CREATE_VARIABLE_NAME(count_adjoint, GET_NAME(DIMENSION_TAG),         \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),   \
                               GET_NAME(BOUNDARY_TAG)));                       \
  this->CREATE_VARIABLE_NAME(h_source_indices_forward,                         \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          source_indices_forward, GET_NAME(DIMENSION_TAG),                     \
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                        \
          GET_NAME(BOUNDARY_TAG)));                                            \
  this->CREATE_VARIABLE_NAME(h_source_indices_backward,                        \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          source_indices_backward, GET_NAME(DIMENSION_TAG),                    \
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                        \
          GET_NAME(BOUNDARY_TAG)));                                            \
  this->CREATE_VARIABLE_NAME(h_source_indices_adjoint,                         \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)) = \
      Kokkos::create_mirror_view(this->CREATE_VARIABLE_NAME(                   \
          source_indices_adjoint, GET_NAME(DIMENSION_TAG),                     \
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),                        \
          GET_NAME(BOUNDARY_TAG)));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ALLOCATE_SOURCES_PER_ELEMENT_TYPE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ALLOCATE_SOURCES_PER_ELEMENT_TYPE

#define ASSIGN_SOURCES_PER_ELEMENT_TYPE(DIMENSION_TAG, MEDIUM_TAG,             \
                                        PROPERTY_TAG, BOUNDARY_TAG)            \
  /* Initialize the index variables */                                         \
  /* I don't think the name needs to be tracked for later use?*/
  int CREATE_VARIABLE_NAME(index_forward, GET_NAME(DIMENSION_TAG),
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                           GET_NAME(BOUNDARY_TAG)) = 0;
  int CREATE_VARIABLE_NAME(index_backward, GET_NAME(DIMENSION_TAG),
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                           GET_NAME(BOUNDARY_TAG)) = 0;
  int CREATE_VARIABLE_NAME(index_adjoint, GET_NAME(DIMENSION_TAG),
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                           GET_NAME(BOUNDARY_TAG)) =
      0; /* Loop over all sources */
  for (int isource = 0; isource < sources.size(); isource++) {
    int ispec = h_elements(isource);
    if ((h_medium_types(ispec) == GET_TAG(MEDIUM_TAG)) &&                    \
        (h_property_types(ispec) == GET_TAG(PROPERTY_TAG)) &&                \
        (h_boundary_types(ispec) == GET_TAG(BOUNDARY_TAG)) {
      if (h_wavefield_types(ispec) ==
          specfem::wavefield::simulation_field::forward) {
        this->CREATE_VARIABLE_NAME(h_element_indices_forward,
                                   GET_NAME(DIMENSION_TAG),
                                   GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                                   GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(
            index_forward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
            GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = ispec;
        this->CREATE_VARIABLE_NAME(h_source_indices_forward,
                                   GET_NAME(DIMENSION_TAG),
                                   GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                                   GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(
            index_forward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
            GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = isource;
          )
          CREATE_VARIABLE_NAME(index_forward, GET_NAME(DIMENSION_TAG),           \
                              GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                              GET_NAME(BOUNDARY_TAG))                           \
          ++;
      } else if (h_wavefield_types(ispec) ==
                 specfem::wavefield::simulation_field::backward) {
          this->CREATE_VARIABLE_NAME(
              h_element_indices_backward, GET_NAME(DIMENSION_TAG),
              GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
              GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(
              index_backward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
              GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = ispec;
          this->CREATE_VARIABLE_NAME(
              h_source_indices_backward, GET_NAME(DIMENSION_TAG),
              GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
              GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(
              index_backward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
              GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = isource;
          CREATE_VARIABLE_NAME(index_backward, GET_NAME(DIMENSION_TAG),
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                               GET_NAME(BOUNDARY_TAG))
          ++;
      } else if (h_wavefield_types(ispec) ==
                 specfem::wavefield::simulation_field::adjoint) {
          this->CREATE_VARIABLE_NAME(
              h_element_indices_adjoint, GET_NAME(DIMENSION_TAG),
              GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
              GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(
              index_adjoint, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
              GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = ispec;
          this->CREATE_VARIABLE_NAME(
              h_source_indices_adjoint, GET_NAME(DIMENSION_TAG),
              GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
              GET_NAME(BOUNDARY_TAG))(CREATE_VARIABLE_NAME(
              index_adjoint, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
              GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))) = isource;
          CREATE_VARIABLE_NAME(index_adjoint, GET_NAME(DIMENSION_TAG),
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),
                               GET_NAME(BOUNDARY_TAG))
          ++;
      }                                                                        \
    }
  }

  /* Copy the data to the device */
  Kokkos::deep_copy(
      this->CREATE_VARIABLE_NAME(
          element_indices_forward, GET_NAME(DIMENSION_TAG),
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),
      this->CREATE_VARIABLE_NAME(h_element_indices_forward,
                                 GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
                                 GET_NAME(PROPERTY_TAG),
                                 GET_NAME(BOUNDARY_TAG)));
  Kokkos::deep_copy(
      this->CREATE_VARIABLE_NAME(
          element_indices_backward, GET_NAME(DIMENSION_TAG),
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),
      this->CREATE_VARIABLE_NAME(h_element_indices_backward,
                                 GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
                                 GET_NAME(PROPERTY_TAG),
                                 GET_NAME(BOUNDARY_TAG)));
  Kokkos::deep_copy(
      this->CREATE_VARIABLE_NAME(
          element_indices_adjoint, GET_NAME(DIMENSION_TAG),
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),
      this->CREATE_VARIABLE_NAME(h_element_indices_adjoint,
                                 GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
                                 GET_NAME(PROPERTY_TAG),
                                 GET_NAME(BOUNDARY_TAG)));
  Kokkos::deep_copy(
      this->CREATE_VARIABLE_NAME(
          source_indices_forward, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),
      this->CREATE_VARIABLE_NAME(h_source_indices_forward,
                                 GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
                                 GET_NAME(PROPERTY_TAG),
                                 GET_NAME(BOUNDARY_TAG)));
  Kokkos::deep_copy(
      this->CREATE_VARIABLE_NAME(
          source_indices_backward, GET_NAME(DIMENSION_TAG),
          GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),
      this->CREATE_VARIABLE_NAME(h_source_indices_backward,
                                 GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
                                 GET_NAME(PROPERTY_TAG),
                                 GET_NAME(BOUNDARY_TAG)));
  Kokkos::deep_copy(
      this->CREATE_VARIABLE_NAME(
          source_indices_adjoint, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
          GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),
      this->CREATE_VARIABLE_NAME(h_source_indices_adjoint,
                                 GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),
                                 GET_NAME(PROPERTY_TAG),
                                 GET_NAME(BOUNDARY_TAG)));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ASSIGN_SOURCES_PER_ELEMENT_TYPE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ASSIGN_SOURCES_PER_ELEMENT_TYPE

  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);
  Kokkos::deep_copy(property_types, h_property_types);
  Kokkos::deep_copy(boundary_types, h_boundary_types);
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::compute::sources::get_sources_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::wavefield::simulation_field wavefield) const {

#define RETURN_VALUE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG)    \
  if ((wavefield == specfem::wavefield::simulation_field::forward) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,      \
                      Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(    \
        CREATE_VARIABLE_NAME(h_element_indices_forward,                        \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),  \
        CREATE_VARIABLE_NAME(h_source_indices_forward,                         \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))); \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::backward) &&         \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,      \
                      Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(    \
        CREATE_VARIABLE_NAME(h_element_indices_backward,                       \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),  \
        CREATE_VARIABLE_NAME(h_source_indices_backward,                        \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))); \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::adjoint) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,      \
                      Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(    \
        CREATE_VARIABLE_NAME(h_element_indices_adjoint,                        \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),  \
        CREATE_VARIABLE_NAME(h_source_indices_adjoint,                         \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG))); \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      RETURN_VALUE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef RETURN_VALUE
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::compute::sources::get_sources_on_device(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::wavefield::simulation_field wavefield) const {

#define RETURN_VALUE(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG, BOUNDARY_TAG)    \
  if ((wavefield == specfem::wavefield::simulation_field::forward) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,      \
                      Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(    \
        CREATE_VARIABLE_NAME(element_indices_forward, GET_NAME(DIMENSION_TAG), \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)),                          \
        CREATE_VARIABLE_NAME(source_indices_forward, GET_NAME(DIMENSION_TAG),  \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)));                         \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::backward) &&         \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,      \
                      Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(    \
        CREATE_VARIABLE_NAME(element_indices_backward,                         \
                             GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),    \
                             GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG)),  \
        CREATE_VARIABLE_NAME(source_indices_backward, GET_NAME(DIMENSION_TAG), \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)));                         \
  }                                                                            \
  if ((wavefield == specfem::wavefield::simulation_field::adjoint) &&          \
      (medium == GET_TAG(MEDIUM_TAG)) &&                                       \
      (property == GET_TAG(PROPERTY_TAG)) &&                                   \
      (boundary == GET_TAG(BOUNDARY_TAG))) {                                   \
    return std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,      \
                      Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >(    \
        CREATE_VARIABLE_NAME(element_indices_adjoint, GET_NAME(DIMENSION_TAG), \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)),                          \
        CREATE_VARIABLE_NAME(source_indices_adjoint, GET_NAME(DIMENSION_TAG),  \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG),     \
                             GET_NAME(BOUNDARY_TAG)));                         \
  }

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      RETURN_VALUE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC) WHERE(
              BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
              BOUNDARY_TAG_STACEY, BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef RETURN_VALUE
}
