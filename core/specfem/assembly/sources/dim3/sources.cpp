#include "specfem/assembly/sources.hpp"
#include "../impl/dim3/source_medium.tpp"
#include "../impl/locate_sources.hpp"
#include "../impl/source_medium.hpp"
#include "algorithms/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

template class specfem::assembly::sources_impl::source_medium<
    specfem::dimension::type::dim3, specfem::element::medium_tag::elastic>;

specfem::assembly::sources<specfem::dimension::type::dim3>::sources(
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim3> > > &sources,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>
        &jacobian_matrix,
    const specfem::assembly::element_types<specfem::dimension::type::dim3>
        &element_types,
    const type_real t0, const type_real dt, const int nsteps)
    : timestep(0), nspec(mesh.nspec),
      element_indices("specfem::sources::elements", sources.size()),
      h_element_indices(Kokkos::create_mirror_view(element_indices)),
      source_indices("specfem::sources::indeces", sources.size()),
      h_source_indices(Kokkos::create_mirror_view(source_indices)),
      medium_types("specfem::sources::medium_types", sources.size()),
      h_medium_types(Kokkos::create_mirror_view(medium_types)),
      property_types("specfem::sources::property_types", sources.size()),
      h_property_types(Kokkos::create_mirror_view(property_types)),
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

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)), CAPTURE(
                                                                      source) {
    auto [sorted_sources, source_indices] =
        specfem::assembly::sources_impl::sort_sources_per_medium<
            _dimension_tag_, _medium_tag_>(sources, element_types, mesh);

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
      assert(element_types.get_medium_tag(ispec) == _medium_tag_);
      h_medium_types(global_isource) = _medium_tag_;
      h_property_types(global_isource) = element_types.get_property_tag(ispec);
      h_boundary_types(global_isource) = element_types.get_boundary_tag(ispec);
      h_wavefield_types(global_isource) = source->get_wavefield_type();
    }

    _source_ = specfem::assembly::sources_impl::source_medium<_dimension_tag_,
                                                              _medium_tag_>(
        sorted_sources, mesh, jacobian_matrix, element_types, t0, dt, nsteps);
  })

  // if the number of sources is not equal to the number of sources
  if (nsources != sources.size()) {
    std::cout << "nsources: " << nsources << std::endl;
    std::cout << "sources.size(): " << sources.size() << std::endl;
    throw std::runtime_error(
        "Not all sources were assigned or sources are assigned multiple times");
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
       BOUNDARY_TAG(NONE)),
      CAPTURE(element_indices_forward, element_indices_backward,
              element_indices_adjoint, source_indices_forward,
              source_indices_backward, source_indices_adjoint,
              h_element_indices_forward, h_element_indices_backward,
              h_element_indices_adjoint, h_source_indices_forward,
              h_source_indices_backward, h_source_indices_adjoint) {
        int count_forward = 0;
        int count_backward = 0;
        int count_adjoint = 0;

        /* Loop over the sources */
        for (int isource = 0; isource < sources.size(); isource++) {
          if ((h_medium_types(isource) == _medium_tag_) &&
              (h_property_types(isource) == _property_tag_) &&
              (h_boundary_types(isource) == _boundary_tag_)) {
            /* Count the number of sources for each wavefield type */
            if (h_wavefield_types(isource) ==
                specfem::wavefield::simulation_field::forward) {
              count_forward++;
            } else if (h_wavefield_types(isource) ==
                       specfem::wavefield::simulation_field::backward) {
              count_backward++;
            } else if (h_wavefield_types(isource) ==
                       specfem::wavefield::simulation_field::adjoint) {
              count_adjoint++;
            }
          }
        }

        /* ==================================== */
        /* Allocating the element specific element_indices array */
        _element_indices_forward_ =
            IndexViewType("specfem::assembly::sources::element_indices_forward",
                          count_forward);
        _element_indices_backward_ = IndexViewType(
            "specfem::assembly::sources::element_indices_backward",
            count_backward);
        _element_indices_adjoint_ =
            IndexViewType("specfem::assembly::sources::element_indices_adjoint",
                          count_adjoint);

        _h_element_indices_forward_ =
            Kokkos::create_mirror_view(_element_indices_forward_);
        _h_element_indices_backward_ =
            Kokkos::create_mirror_view(_element_indices_backward_);
        _h_element_indices_adjoint_ =
            Kokkos::create_mirror_view(_element_indices_adjoint_);

        /* ==================================== */
        /* Allocation the element specific source_indices arrays. */
        /* We do not need a separate counter for this as it is the same */
        /* as the count for the element_indices */
        _source_indices_forward_ =
            IndexViewType("specfem::assembly::sources::source_indices_forward",
                          count_forward);
        _source_indices_backward_ =
            IndexViewType("specfem::assembly::sources::source_indices_backward",
                          count_backward);
        _source_indices_adjoint_ =
            IndexViewType("specfem::assembly::sources::source_indices_adjoint",
                          count_adjoint);

        _h_source_indices_forward_ =
            Kokkos::create_mirror_view(_source_indices_forward_);
        _h_source_indices_backward_ =
            Kokkos::create_mirror_view(_source_indices_backward_);
        _h_source_indices_adjoint_ =
            Kokkos::create_mirror_view(_source_indices_adjoint_);

        /* Initialize the index variables */
        int index_forward = 0;
        int index_backward = 0;
        int index_adjoint = 0;
        /* Loop over all sources */
        for (int isource = 0; isource < sources.size(); isource++) {
          int ispec = h_element_indices(isource);
          if ((h_medium_types(isource) == _medium_tag_) &&
              (h_property_types(isource) == _property_tag_) &&
              (h_boundary_types(isource) == _boundary_tag_)) {
            if (h_wavefield_types(isource) ==
                specfem::wavefield::simulation_field::forward) {
              /* Assign global ispec to local forward element index array */
              /* h_element_indices_forward_<dim>_<medium>_<property> = ispec*/
              _h_element_indices_forward_(index_forward) = ispec;
              /* Assign global forward source index to local source index */
              /* h_source_indices_forward_<dim>_<medium>_<property> = isource */
              _h_source_indices_forward_(index_forward) = isource;
              /* Increase forward counter index*/
              index_forward++;
            } else if (h_wavefield_types(isource) ==
                       specfem::wavefield::simulation_field::backward) {
              /* Assign global ispec to local backward element index array */
              /* h_element_indices_backward_<dim>_<medium>_<property> = ispec */
              _h_element_indices_backward_(index_backward) = ispec;
              /* Assign global backward source index to local source index */
              /* h_source_indices_backward_<dim>_<medium>_<property> = isource
               */
              _h_source_indices_backward_(index_backward) = isource;
              index_backward++;
            } else if (h_wavefield_types(isource) ==
                       specfem::wavefield::simulation_field::adjoint) {
              /* Assign global ispec to local adjoint element index array */
              /* h_element_indices_adjoint_<dim>_<medium>_<property> = ispec */
              _h_element_indices_adjoint_(index_adjoint) = ispec;
              _h_source_indices_adjoint_(index_adjoint) = isource;
              index_adjoint++;
            }
          }
        }

        Kokkos::deep_copy(_element_indices_forward_,
                          _h_element_indices_forward_);
        Kokkos::deep_copy(_element_indices_backward_,
                          _h_element_indices_backward_);
        Kokkos::deep_copy(_element_indices_adjoint_,
                          _h_element_indices_adjoint_);
        Kokkos::deep_copy(_source_indices_forward_, _h_source_indices_forward_);
        Kokkos::deep_copy(_source_indices_backward_,
                          _h_source_indices_backward_);
        Kokkos::deep_copy(_source_indices_adjoint_, _h_source_indices_adjoint_);
      })

  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);
  Kokkos::deep_copy(property_types, h_property_types);
  Kokkos::deep_copy(boundary_types, h_boundary_types);
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::sources<specfem::dimension::type::dim3>::get_sources_on_host(
    const specfem::element::medium_tag medium,
    const specfem::element::property_tag property,
    const specfem::element::boundary_tag boundary,
    const specfem::wavefield::simulation_field wavefield) const {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
       BOUNDARY_TAG(NONE)),
      CAPTURE(h_element_indices_forward, h_element_indices_backward,
              h_element_indices_adjoint, h_source_indices_forward,
              h_source_indices_backward, h_source_indices_adjoint) {
        if ((wavefield == specfem::wavefield::simulation_field::forward) &&
            (medium == _medium_tag_) && (property == _property_tag_) &&
            (boundary == _boundary_tag_)) {
          return std::make_tuple(_h_element_indices_forward_,
                                 _h_source_indices_forward_);
        } else if ((wavefield ==
                    specfem::wavefield::simulation_field::backward) &&
                   (medium == _medium_tag_) && (property == _property_tag_) &&
                   (boundary == _boundary_tag_)) {
          return std::make_tuple(_h_element_indices_backward_,
                                 _h_source_indices_backward_);
        } else if ((wavefield ==
                    specfem::wavefield::simulation_field::adjoint) &&
                   (medium == _medium_tag_) && (property == _property_tag_) &&
                   (boundary == _boundary_tag_)) {
          return std::make_tuple(_h_element_indices_adjoint_,
                                 _h_source_indices_adjoint_);
        }
      })

  Kokkos::abort("No sources found for the given parameters. Please check the "
                "input parameters and try again.");
  return std::make_tuple(
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>(),
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>());
}

// This function is crucial for the computing the source contribution
// to the wavefield. It returns the global indices of the relevant elements
// and the source indices for the wavefield type.
std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::assembly::sources<specfem::dimension::type::dim3>::
    get_sources_on_device(
        const specfem::element::medium_tag medium,
        const specfem::element::property_tag property,
        const specfem::element::boundary_tag boundary,
        const specfem::wavefield::simulation_field wavefield) const {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
       BOUNDARY_TAG(NONE)),
      CAPTURE(element_indices_forward, element_indices_backward,
              element_indices_adjoint, source_indices_forward,
              source_indices_backward, source_indices_adjoint) {
        if ((wavefield == specfem::wavefield::simulation_field::forward) &&
            (medium == _medium_tag_) && (property == _property_tag_) &&
            (boundary == _boundary_tag_)) {
          return std::make_tuple(_element_indices_forward_,
                                 _source_indices_forward_);
        } else if ((wavefield ==
                    specfem::wavefield::simulation_field::backward) &&
                   (medium == _medium_tag_) && (property == _property_tag_) &&
                   (boundary == _boundary_tag_)) {
          return std::make_tuple(_element_indices_backward_,
                                 _source_indices_backward_);
        } else if ((wavefield ==
                    specfem::wavefield::simulation_field::adjoint) &&
                   (medium == _medium_tag_) && (property == _property_tag_) &&
                   (boundary == _boundary_tag_)) {
          return std::make_tuple(_element_indices_adjoint_,
                                 _source_indices_adjoint_);
        }
      })

  Kokkos::abort("No sources found for the given parameters. Please check the "
                "input parameters and try again.");
  return std::make_tuple(Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(),
                         Kokkos::View<int *, Kokkos::DefaultExecutionSpace>());
}
