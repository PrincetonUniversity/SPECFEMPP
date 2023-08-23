#ifndef _DOMAIN_TPP
#define _DOMAIN_TPP

#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

template <typename element_type>
static using element_container =
    typename specfem::domain::impl::elements::container<element_type>;

template <class dimension, class medium, class qp_type, class... traits>
static using element_type =
    typename specfem::domain::impl::elements::element<dimension, medium,
                                                      qp_type, traits...>;

template <typename source_type>
static using source_container =
    typename specfem::domain::impl::sources::container<source_type>;

template <class dimension, class medium, class qp_type, class... traits>
static using source_type =
    typename specfem::domain::impl::sources::source<dimension, medium, qp_type,
                                                    traits...>;

template <typename receiver_type>
static using receiver_container =
    typename specfem::domain::impl::receivers::container<receiver_type>;

template <class dimension, class medium, class qp_type, class... traits>
static using receiver_type =
    typename specfem::domain::impl::receivers::receiver<dimension, medium,
                                                        qp_type, traits...>;

template <class medium>
void initialize_views(
    const int &nglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {

  constexpr int components = medium::components;

  Kokkos::parallel_for(
      "specfem::Domain::acoustic::initiaze_views",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_LAMBDA(const int iglob, const int idim) {
        field(iglob, idim) = 0;
        field_dot(iglob, idim) = 0;
        field_dot_dot(iglob, idim) = 0;
        rmass_inverse(iglob, idim) = 0;
      });
}

template <class medium, class qp_type>
void initialize_rmass_inverse(
    const specfem::kokkos::DeviceView3d<int> ibool,
    const specfem::kokkos::DeviceView1d<type_real> wxgll,
    const specfem::kokkos::DeviceView1d<type_real> wzgll,
    const specfem::kokkos::DeviceView1d<element_container<element_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &elements,
    const qp_type &quadrature_points,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {
  // Compute the mass matrix

  constexpr int components = medium::components;
  constexpr auto value = medium::value;

  const auto ispec_type = properties.ispec_type;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  const int ngllxz = ngllx * ngllz;

  const int nglob = rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::domain::domain::rmass_matrix",
      specfem::kokkos::DeviceTeam(elements.extent(0), Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto element = this->elements(team_member.league_rank());
        const auto ispec = element.get_ispec();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec, iz, ix);

              type_real[components] mass_matrix_element =
                  wxgll(ix) * wzgll(iz) *
                  element.compute_mass_matrix_component(xz);

              for (int icomponent = 0; icomponent < components; icomponent++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&rmass_inverse(iglob, icomponent),
                                     mass_matrix_element[icomponent]);
                });
              }
            });
      });
}

template <class medium, class qp_type>
void assign_elemental_properties(
    specfem::compute::partial_derivatives &partial_derivatives,
    specfem::compute::properties &properties,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView1d<element_container<element_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &elements,
    const int &nspec, const int &ngllz, const int &ngllx, int &nelem_domain) {

  const auto value = medium::value;
  const auto components = medium::components;

  // count number of elements in this domain
  nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == value) {
      nelem_domain++;
    }
  }

  specfem::kokkos::DeviceView1d<int> ispec_domain(
      "specfem::domain::domain::h_ispec_domain", nelem_domain);
  specfem::kokkos::HostMirror1d<int> h_ispec_domain =
      Kokkos::create_mirror_view(ispec_domain);

  elements = specfem::kokkos::DeviceView1d<
      acoustic_tmp::element_container<acoustic_tmp::element_type<qp_type> > >(
      "specfem::domain::domain::elements", nelem_domain);

  // Get ispec for each element in this domain
  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == specfem::enums::element::acoustic) {
      h_ispec_domain(index) = ispec;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index == nelem_domain);
#endif

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  // Assign elemental properties
  // ----------------------------------------------------------------------
  auto h_elements = Kokkos::create_mirror_view(elements);

  Kokkos::parallel_for(
      "specfem::domain::allocate_memory",
      specfem::kokkos::HostRange(0, h_elements.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        h_elements(i).element = (element_type<qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                element_type<qp_type,
                             specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(elements, h_elements);

  Kokkos::parallel_for(
      "specfem::domain::instantialize_element",
      specfem::kokkos::DeviceRange(0, ispec_domain.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        const int ispec = ispec_domain(i);
        auto &element = elements(ispec).element;
        new (element)
            element_type<qp_type, specfem::enums::element::property::isotropic>(
                ispec, partial_derivatives, properties);
      });

  Kokkos::fence();
};

template <class medium, class qp_type>
void assign_source_properties(
    specfem::compute::partial_derivatives &partial_derivatives,
    specfem::compute::properties &compute_sources,
    specfem::kokkos::DeviceView1d<source_container<source_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &sources) {

  const auto value = medium::value;
  const auto h_ispec_type = properties.h_ispec_type;
  const auto ispec_array = compute_sources.ispec_array;
  const auto h_ispec_array = compute_sources.h_ispec_array;

  int nsources_domain = 0;

  // Count the number of sources in the domain
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    if (h_ispec_type(h_ispec_array(isource)) == value) {
      nsources_domain++;
    }
  }

  sources =
      specfem::kokkos::DeviceView1d<source_container<source_type<qp_type> > >(
          "specfem::domain::domain::sources", nsources_domain);

  specfem::kokkos::DeviceView1d<int> my_sources(
      "specfem::domain::domain::my_sources", nsources_domain);

  auto h_my_sources = Kokkos::create_mirror_view(my_sources);
  auto h_sources = Kokkos::create_mirror_view(sources);

  // Check if the source is in the domain
  int index = 0;
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    if (h_ispec_type(h_ispec_array(isource)) == value) {
      h_my_sources(index) = isource;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index == nsources_domain);
#endif

  Kokkos::deep_copy(my_sources, h_my_sources);

  // Allocate memory for the sources on the device
  Kokkos::parallel_for(
      "specfem::domain::domain::allocate_memory",
      specfem::kokkos::HostRange(0, nsources_domain),
      KOKKOS_LAMBDA(const int isource) {
        h_sources(isource).source = (source_type<qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                source_type<qp_type,
                            specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(sources, h_sources);

  // Initialize the sources
  Kokkos::parallel_for(
      "specfem::domain::domain::initialize_source",
      specfem::kokkos::DeviceRange(0, nsources_domain),
      KOKKOS_LAMBDA(const int isource) {
        const int ispec = ispec_array(my_sources(isource));

        specfem::forcing_function::stf *source_time_function =
            compute_sources.stf_array(my_sources(isource)).T;

        const auto sv_source_array =
            Kokkos::subview(compute_sources.source_array, my_sources(isource),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

        auto &source = sources(isource).source;
        new (source)
            source_type<qp_type, specfem::enums::element::property::isotropic>(
                ispec, properties, sv_source_array, source_time_function);
      });

  Kokkos::fence();
  return;
}

template <class medium, class qp_type>
void initialize_receivers(
    const specfem::compute::receivers &compute_receivers,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties,
    specfem::kokkos::DeviceView1d<receiver_container<receiver_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &receivers) {

  const auto value = medium::value;
  const auto h_ispec_type = properties.h_ispec_type;
  const auto ispec_array = compute_receivers.ispec_array;
  const auto h_ispec_array = compute_receivers.h_ispec_array;
  const auto seis_types = compute_receivers.seismogram_types;

  int nreceivers_domain = 0;

  // Count the number of receivers in the domain
  for (int irec = 0; irec < ispec_array.extent(0); irec++) {
    if (h_ispec_type(h_ispec_array(irec)) == value) {
      nreceivers_domain++;
    }
  }

  nreceivers_domain = nreceivers_domain * seis_types.extent(0);

  receivers = specfem::kokkos::DeviceView1d<
      receiver_container<receiver_type<qp_type> > >(
      "specfem::domain::domain::receivers", nreceivers_domain);

  specfem::kokkos::DeviceView1d<int> my_receivers(
      "specfem::domain::domain::my_receivers", nreceivers_domain);

  auto h_my_receivers = Kokkos::create_mirror_view(my_receivers);
  auto h_receivers = Kokkos::create_mirror_view(receivers);

  // Check if the receiver is in the domain
  int index = 0;
  for (int irec = 0; irec < ispec_array.extent(0); irec++) {
    if (h_ispec_type(h_ispec_array(irec)) ==
        specfem::enums::element::acoustic) {
      h_my_receivers(index) = irec;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index * seis_types.extent(0) == nreceivers_domain);
#endif

  Kokkos::deep_copy(my_receivers, h_my_receivers);

  // Allocate memory for the receivers on the device
  Kokkos::parallel_for(
      "specfem::domain::domain::allocate_memory",
      specfem::kokkos::HostRange(0, nreceivers_domain),
      KOKKOS_LAMBDA(const int irec) {
        h_receivers(irec).receiver = (receiver_type<qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                receiver_type<qp_type,
                              specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(receivers, h_receivers);

  Kokkos::parallel_for(
      "specfem::domain::domain::initialize_receiver",
      specfem::kokkos::DeviceRange(0, nreceivers_domain),
      KOKKOS_LAMBDA(const int inum) {
        const int irec = my_receivers(inum / seis_types.extent(0));
        const int iseis = inum % seis_types.extent(0);
        const int ispec = ispec_array(irec);
        const auto seis_type = seis_types(iseis);
        const type_real cos_rec = compute_receivers.cos_recs(irec);
        const type_real sin_rec = compute_receivers.sin_recs(irec);

        auto sv_receiver_array =
            Kokkos::subview(compute_receivers.receiver_array, irec, Kokkos::ALL,
                            Kokkos::ALL, Kokkos::ALL);

        // Subview the seismogram array at current receiver and current
        // seismogram value
        auto sv_receiver_seismogram =
            Kokkos::subview(compute_receivers.seismogram, Kokkos::ALL, iseis,
                            irec, Kokkos::ALL);
        auto sv_receiver_field =
            Kokkos::subview(compute_receivers.receiver_field, Kokkos::ALL, irec,
                            iseis, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

        auto &receiver = receivers(inum).receiver;
        new (receiver)
            receiver_type<qp_type,
                          specfem::enums::element::property::isotropic>(
                ispec, sin_rec, cos_rec, seis_type, sv_receiver_array,
                sv_receiver_seismogram, partial_derivatives, properties,
                sv_receiver_field);
      });

  return;
}

template <class medium, class qp_type>
specfem::domain::domain<medium, qp_type>::domain(
    const int ndim, const int nglob, const qp_type &quadrature_points,
    specfem::compute::compute *compute,
    specfem::compute::properties material_properties,
    specfem::compute::partial_derivatives partial_derivatives,
    specfem::compute::sources compute_sources,
    specfem::compute::receivers compute_receivers,
    specfem::quadrature::quadrature *quadx,
    specfem::quadrature::quadrature *quadz)
    : field(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::acoustic::field", nglob,
          specfem::enums::element::medium::acoustic::components)),
      field_dot(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::acoustic::field_dot", nglob,
          specfem::enums::element::medium::acoustic::components)),
      field_dot_dot(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::acoustic::field_dot_dot", nglob,
              specfem::enums::element::medium::acoustic::components)),
      rmass_inverse(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::acoustic::rmass_inverse", nglob,
              specfem::enums::element::medium::acoustic::components)),
      quadrature_points(quadrature_points), compute(compute), quadx(quadx),
      quadz(quadz) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  //----------------------------------------------------------------------------
  // Initialize views

  initialize_views<medium>(nglob, this->field, this->field_dot,
                           this->field_dot_dot, this->rmass_inverse);

  // ----------------------------------------------------------------------------
  // Inverse of mass matrix

  assign_elemental_properties(partial_derivatives, material_properties,
                              this->field_dot_dot, this->elements, nspec, ngllz,
                              ngllx, this->nelem_domain);

  //----------------------------------------------------------------------------
  // Inverse of mass matrix

  initialize_rmass_inverse(compute->ibool, quadx->get_w(), quadz->get_w(),
                           this->elements, this->quadrature_points,
                           this->rmass_inverse);

  // ----------------------------------------------------------------------------
  // Initialize the sources

  initialize_sources(material_properties, compute_sources, this->sources);

  // ----------------------------------------------------------------------------
  // Initialize the receivers
  initialize_receivers(compute_receivers, partial_derivatives,
                       material_properties, this->receivers);

  return;
};

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_field(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::acoustic,
                             qp_type>::sync_field_dot(specfem::sync::kind
                                                          kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot, field_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot, h_field_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_field_dot_dot(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_rmass_inverse(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::divide_mass_matrix() {

  constexpr int components = medium::components;
  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::acoustic::divide_mass_matrix",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        this->field_dot_dot(iglob, idim) =
            this->field_dot_dot(iglob, idim) * this->rmass_inverse(iglob, idim);
      });

  Kokkos::fence();

  return;
}

#endif
